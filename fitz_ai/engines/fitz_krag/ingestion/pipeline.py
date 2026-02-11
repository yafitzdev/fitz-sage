# fitz_ai/engines/fitz_krag/ingestion/pipeline.py
"""
KRAG Ingestion Pipeline.

Scans source files, extracts symbols via language strategies, generates
LLM summaries in batches, embeds summaries, and stores everything in
PostgreSQL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.ingestion.import_graph_store import ImportGraphStore
from fitz_ai.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
from fitz_ai.engines.fitz_krag.ingestion.schema import ensure_schema
from fitz_ai.engines.fitz_krag.ingestion.section_store import SectionStore
from fitz_ai.engines.fitz_krag.ingestion.strategies.base import IngestResult, SymbolEntry
from fitz_ai.engines.fitz_krag.ingestion.strategies.python_code import (
    PythonCodeIngestStrategy,
)
from fitz_ai.engines.fitz_krag.ingestion.strategies.technical_doc import (
    DOC_EXTENSIONS,
    DocIngestResult,
    SectionEntry,
    TechnicalDocIngestStrategy,
)
from fitz_ai.engines.fitz_krag.ingestion.symbol_store import SymbolStore

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.llm.providers.base import ChatProvider, EmbeddingProvider
    from fitz_ai.storage.postgres import PostgresConnectionManager

logger = logging.getLogger(__name__)

# Extensions handled by code strategies
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
}


class KragIngestPipeline:
    """
    Full ingestion pipeline for KRAG engine.

    Steps:
    1. Scan source for files
    2. Compute content hashes, diff against stored hashes
    3. For new/changed files: extract -> summarize -> embed -> store
    4. For deleted files: cascade delete
    """

    def __init__(
        self,
        config: "FitzKragConfig",
        chat: "ChatProvider",
        embedder: "EmbeddingProvider",
        connection_manager: "PostgresConnectionManager",
        collection: str,
    ):
        self._config = config
        self._chat = chat
        self._embedder = embedder
        self._cm = connection_manager
        self._collection = collection

        # Stores
        self._raw_store = RawFileStore(connection_manager, collection)
        self._symbol_store = SymbolStore(connection_manager, collection)
        self._import_store = ImportGraphStore(connection_manager, collection)
        self._section_store = SectionStore(connection_manager, collection)

        # Code strategies
        self._strategies: dict[str, Any] = {}
        if "python" in config.code_languages:
            self._strategies["python"] = PythonCodeIngestStrategy()

        # Document strategy
        self._doc_strategy = TechnicalDocIngestStrategy()

        # Ensure schema
        ensure_schema(connection_manager, collection, embedder.dimensions)

    def ingest(self, source: Path) -> dict[str, Any]:
        """
        Run the full ingestion pipeline.

        Args:
            source: Path to source directory or single file

        Returns:
            Stats dict: files_scanned, files_new, files_changed, files_deleted,
                        symbols_extracted, symbols_embedded
        """
        source = Path(source)
        stats = {
            "files_scanned": 0,
            "files_new": 0,
            "files_changed": 0,
            "files_deleted": 0,
            "symbols_extracted": 0,
            "symbols_embedded": 0,
            "sections_extracted": 0,
            "sections_embedded": 0,
            "collection": self._collection,
        }

        # 1. Scan files
        file_paths = self._scan_files(source)
        stats["files_scanned"] = len(file_paths)

        # 2. Compute hashes and diff
        existing_hashes = self._raw_store.list_hashes()
        existing_ids = self._raw_store.list_ids_by_path()
        current_paths: set[str] = set()

        new_files: list[tuple[str, Path]] = []  # (relative_path, absolute_path)
        changed_files: list[tuple[str, Path]] = []

        for abs_path in file_paths:
            rel_path = self._relative_path(abs_path, source)
            current_paths.add(rel_path)
            content_hash = _hash_file(abs_path)

            if rel_path not in existing_hashes:
                new_files.append((rel_path, abs_path))
            elif existing_hashes[rel_path] != content_hash:
                changed_files.append((rel_path, abs_path))

        # 3. Process new/changed files
        all_symbols: list[SymbolEntry] = []
        all_symbol_file_ids: list[str] = []
        all_import_edges: list[dict[str, Any]] = []
        all_sections: list[SectionEntry] = []
        all_section_file_ids: list[str] = []

        for rel_path, abs_path in new_files + changed_files:
            file_id = existing_ids.get(rel_path, str(uuid.uuid4()))
            ext = abs_path.suffix.lower()

            if ext in EXTENSION_MAP:
                result = self._process_code_file(rel_path, abs_path, file_id)
                if result:
                    symbols, import_edges = result
                    all_symbols.extend(symbols)
                    all_symbol_file_ids.extend([file_id] * len(symbols))
                    all_import_edges.extend(import_edges)
            elif ext in DOC_EXTENSIONS:
                sections = self._process_doc_file(rel_path, abs_path, file_id)
                if sections:
                    all_sections.extend(sections)
                    all_section_file_ids.extend([file_id] * len(sections))

        stats["files_new"] = len(new_files)
        stats["files_changed"] = len(changed_files)
        stats["symbols_extracted"] = len(all_symbols)
        stats["sections_extracted"] = len(all_sections)

        # 4a. Batch summarize + embed symbols
        if all_symbols:
            summaries = self._summarize_symbols(all_symbols)
            vectors = self._embed_summaries(summaries)

            # Store symbols with summaries and vectors
            symbol_dicts = []
            for i, sym in enumerate(all_symbols):
                symbol_dicts.append(
                    {
                        "id": str(uuid.uuid4()),
                        "name": sym.name,
                        "qualified_name": sym.qualified_name,
                        "kind": sym.kind,
                        "raw_file_id": all_symbol_file_ids[i],
                        "start_line": sym.start_line,
                        "end_line": sym.end_line,
                        "signature": sym.signature,
                        "summary": summaries[i] if i < len(summaries) else None,
                        "summary_vector": vectors[i] if i < len(vectors) else None,
                        "imports": sym.imports,
                        "references": sym.references,
                        "metadata": {},
                    }
                )
            self._symbol_store.upsert_batch(symbol_dicts)
            stats["symbols_embedded"] = len(vectors)

        # Store import edges
        if all_import_edges:
            self._import_store.upsert_batch(all_import_edges)

        # 4b. Batch summarize + embed sections
        if all_sections:
            section_summaries = self._summarize_sections(all_sections)
            section_vectors = self._embed_summaries(section_summaries)

            section_dicts = []
            for i, sec in enumerate(all_sections):
                section_dicts.append(
                    {
                        "id": str(uuid.uuid4()),
                        "raw_file_id": all_section_file_ids[i],
                        "title": sec.title,
                        "level": sec.level,
                        "page_start": sec.page_start,
                        "page_end": sec.page_end,
                        "content": sec.content,
                        "summary": (section_summaries[i] if i < len(section_summaries) else None),
                        "summary_vector": (
                            section_vectors[i] if i < len(section_vectors) else None
                        ),
                        "parent_section_id": sec.parent_id,
                        "position": sec.position,
                        "metadata": sec.metadata,
                    }
                )
            self._section_store.upsert_batch(section_dicts)
            stats["sections_embedded"] = len(section_vectors)

        # 5. Delete removed files
        deleted_paths = set(existing_hashes.keys()) - current_paths
        for del_path in deleted_paths:
            if del_path in existing_ids:
                self._raw_store.delete(existing_ids[del_path])
        stats["files_deleted"] = len(deleted_paths)

        logger.info(
            f"KRAG ingest complete: {stats['files_scanned']} scanned, "
            f"{stats['files_new']} new, {stats['files_changed']} changed, "
            f"{stats['symbols_extracted']} symbols, "
            f"{stats['sections_extracted']} sections"
        )
        return stats

    def _scan_files(self, source: Path) -> list[Path]:
        """Scan source for files matching enabled code + document strategies."""
        extensions = set()
        for lang in self._config.code_languages:
            for ext, lang_name in EXTENSION_MAP.items():
                if lang_name == lang:
                    extensions.add(ext)

        # Include document extensions
        extensions.update(DOC_EXTENSIONS)

        if source.is_file():
            if source.suffix.lower() in extensions:
                return [source]
            return []

        files = []
        for ext in extensions:
            files.extend(source.rglob(f"*{ext}"))

        # Filter out common non-source directories
        skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", ".tox", ".eggs"}
        return [f for f in sorted(files) if not any(part in skip_dirs for part in f.parts)]

    def _relative_path(self, abs_path: Path, source: Path) -> str:
        """Get relative path string."""
        try:
            return str(abs_path.relative_to(source)).replace("\\", "/")
        except ValueError:
            return str(abs_path).replace("\\", "/")

    def _process_code_file(
        self, rel_path: str, abs_path: Path, file_id: str
    ) -> tuple[list[SymbolEntry], list[dict[str, Any]]] | None:
        """Process a code file: store raw + extract symbols/imports."""
        ext = abs_path.suffix.lower()
        lang = EXTENSION_MAP.get(ext)
        if not lang or lang not in self._strategies:
            return None

        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Cannot read {abs_path}: {e}")
            return None

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Store raw file
        self._raw_store.upsert(
            file_id=file_id,
            path=rel_path,
            content=content,
            content_hash=content_hash,
            file_type=ext,
            size_bytes=len(content.encode()),
        )

        # Extract symbols
        strategy = self._strategies[lang]
        result: IngestResult = strategy.extract(content, rel_path)

        # Delete old symbols for this file (will be replaced)
        self._symbol_store.delete_by_file(file_id)
        self._import_store.delete_by_file(file_id)

        # Build import edges
        import_edges = [
            {
                "source_file_id": file_id,
                "target_module": imp.target_module,
                "target_file_id": None,  # Resolved later if needed
                "import_names": imp.import_names,
            }
            for imp in result.imports
        ]

        return result.symbols, import_edges

    def _process_doc_file(
        self, rel_path: str, abs_path: Path, file_id: str
    ) -> list[SectionEntry] | None:
        """Process a document file: store raw + parse + extract sections."""
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Cannot read {abs_path}: {e}")
            return None

        ext = abs_path.suffix.lower()
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Store raw file
        self._raw_store.upsert(
            file_id=file_id,
            path=rel_path,
            content=content,
            content_hash=content_hash,
            file_type=ext,
            size_bytes=len(content.encode()),
        )

        # Parse document to get structured elements
        parsed_doc = self._parse_document(abs_path)
        if not parsed_doc:
            return None

        # Extract sections from parsed document
        result: DocIngestResult = self._doc_strategy.extract(parsed_doc, rel_path)
        if not result.sections:
            return None

        # Delete old sections for this file (will be replaced)
        self._section_store.delete_by_file(file_id)

        return result.sections

    def _parse_document(self, abs_path: Path) -> Any:
        """Parse a document file using the ingestion parser router."""
        try:
            from fitz_ai.ingestion.parser.router import ParserRouter
            from fitz_ai.ingestion.source.base import SourceFile

            router = ParserRouter()
            source_file = SourceFile(
                uri=f"file://{abs_path}",
                local_path=abs_path,
            )
            return router.parse(source_file)
        except Exception as e:
            logger.warning(f"Document parsing failed for {abs_path}: {e}")
            return None

    def _summarize_symbols(self, symbols: list[SymbolEntry]) -> list[str]:
        """Generate 1-2 sentence summaries for symbols, batched."""
        summaries: list[str] = []
        batch_size = self._config.summary_batch_size

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            prompt = self._build_summary_prompt(batch)

            try:
                response = self._chat.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You summarize code symbols. For each symbol, write a concise "
                                "1-2 sentence description of what it does. Return a JSON array "
                                "of strings, one per symbol, in the same order."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
                batch_summaries = self._parse_summary_response(response, len(batch))
            except Exception as e:
                logger.warning(f"Summary generation failed for batch: {e}")
                batch_summaries = [f"{sym.kind} {sym.name}" for sym in batch]
            summaries.extend(batch_summaries)

        return summaries

    def _build_summary_prompt(self, batch: list[SymbolEntry]) -> str:
        """Build prompt for batch summarization."""
        parts = []
        for i, sym in enumerate(batch):
            # Truncate source to avoid token overflow
            source = sym.source[:500] if sym.source else "(no source)"
            parts.append(
                f"Symbol {i + 1}: {sym.kind} '{sym.qualified_name}'\n"
                f"Signature: {sym.signature or 'N/A'}\n"
                f"Source:\n```\n{source}\n```"
            )
        return "\n\n".join(parts)

    def _parse_summary_response(self, response: str, expected_count: int) -> list[str]:
        """Parse LLM response into list of summary strings."""
        # Try JSON array first
        try:
            # Extract JSON from possible markdown code block
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) >= expected_count:
                return [str(s) for s in parsed[:expected_count]]
        except (json.JSONDecodeError, IndexError):
            pass

        # Fallback: split by numbered lines
        lines = [
            line.strip()
            for line in response.strip().splitlines()
            if line.strip() and not line.strip().startswith("```")
        ]
        # Strip leading numbers like "1. " or "1: "
        cleaned = []
        for line in lines:
            for prefix_len in range(1, 4):
                if len(line) > prefix_len + 2 and line[prefix_len] in ".):":
                    line = line[prefix_len + 1 :].strip()
                    break
            cleaned.append(line)

        if len(cleaned) >= expected_count:
            return cleaned[:expected_count]

        # Pad if needed
        return cleaned + ["(no summary)"] * (expected_count - len(cleaned))

    def _summarize_sections(self, sections: list[SectionEntry]) -> list[str]:
        """Generate 1-2 sentence summaries for document sections, batched."""
        summaries: list[str] = []
        batch_size = self._config.summary_batch_size

        for i in range(0, len(sections), batch_size):
            batch = sections[i : i + batch_size]
            prompt = self._build_section_summary_prompt(batch)

            try:
                response = self._chat.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You summarize document sections. For each section, write a "
                                "concise 1-2 sentence description of its content. Return a "
                                "JSON array of strings, one per section, in the same order."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
                batch_summaries = self._parse_summary_response(response, len(batch))
            except Exception as e:
                logger.warning(f"Section summary generation failed for batch: {e}")
                batch_summaries = [sec.title for sec in batch]
            summaries.extend(batch_summaries)

        return summaries

    def _build_section_summary_prompt(self, batch: list[SectionEntry]) -> str:
        """Build prompt for section batch summarization."""
        parts = []
        for i, sec in enumerate(batch):
            # Truncate content to avoid token overflow
            content = sec.content[:800] if sec.content else "(no content)"
            parts.append(
                f"Section {i + 1}: '{sec.title}' (level {sec.level})\n" f"Content:\n{content}"
            )
        return "\n\n".join(parts)

    def _embed_summaries(self, summaries: list[str]) -> list[list[float]]:
        """Embed summary strings using the configured embedder."""
        if not summaries:
            return []
        try:
            return self._embedder.embed_batch(summaries)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return []


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()
