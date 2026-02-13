# fitz_ai/engines/fitz_krag/retrieval/strategies/agentic_search.py
"""
Agentic search strategy — LLM-driven file selection from manifest.

For files not yet indexed (not at EMBEDDED state), this strategy:
1. Builds compact manifest text
2. Uses BM25 pre-filter when >50 unindexed files
3. Asks LLM to pick relevant files
4. Reads content from disk
5. Returns standard Address objects
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.engines.fitz_krag.progressive.manifest import (
        FileManifest,
        ManifestEntry,
    )

logger = logging.getLogger(__name__)

# Max unindexed files to send to LLM
_BM25_PREFILTER_THRESHOLD = 50
_LLM_MAX_FILES = 10


class AgenticSearchStrategy:
    """LLM-driven file selection from manifest for unindexed files."""

    def __init__(
        self,
        manifest: "FileManifest",
        source_dir: Path,
        chat_factory: Any,
        config: "FitzKragConfig",
    ) -> None:
        self._manifest = manifest
        self._source_dir = source_dir
        self._chat_factory = chat_factory
        self._config = config

    def retrieve(self, query: str, limit: int) -> list[Address]:
        """Retrieve addresses for unindexed files via LLM file selection.

        1. Get files NOT at EMBEDDED state from manifest
        2. Build compact manifest text (~50-100 tokens/file)
        3. If >50 unindexed files: BM25 pre-filter to top 50
        4. LLM picks ~5-10 candidate files
        5. Read file content from disk
        6. Create Address objects with AST line ranges from manifest
        """
        from fitz_ai.engines.fitz_krag.progressive.manifest import FileState

        unindexed = self._manifest.files_not_in_state(FileState.EMBEDDED)
        if not unindexed:
            return []

        # BM25 pre-filter if too many files
        if len(unindexed) > _BM25_PREFILTER_THRESHOLD:
            unindexed = self._bm25_prefilter(unindexed, query)

        # Build manifest text for LLM
        manifest_text = self._manifest.to_manifest_text(unindexed)

        # LLM selects files
        selected_paths = self._llm_select_files(manifest_text, query)
        if not selected_paths:
            return []

        # Map selected paths to entries
        entries_map = {e.rel_path: e for e in unindexed}
        selected_entries = [
            entries_map[p] for p in selected_paths if p in entries_map
        ]

        # Create addresses from selected files
        addresses: list[Address] = []
        for entry in selected_entries[:limit]:
            content = self._read_from_disk(entry)
            if content is None:
                continue

            entry_addresses = self._create_addresses(entry, content)
            addresses.extend(entry_addresses)

        return addresses

    def _bm25_prefilter(
        self, entries: list["ManifestEntry"], query: str
    ) -> list["ManifestEntry"]:
        """In-memory token overlap scoring (pure Python, no deps).

        Tokenize query + manifest text, score by term frequency, return top 50.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return entries[:_BM25_PREFILTER_THRESHOLD]

        query_tf = Counter(query_tokens)
        scored: list[tuple[float, "ManifestEntry"]] = []

        for entry in entries:
            doc_text = _entry_to_text(entry)
            doc_tokens = _tokenize(doc_text)
            if not doc_tokens:
                scored.append((0.0, entry))
                continue

            doc_tf = Counter(doc_tokens)
            doc_len = len(doc_tokens)

            # Simple BM25-like scoring
            score = 0.0
            for term, qf in query_tf.items():
                tf = doc_tf.get(term, 0)
                if tf > 0:
                    # TF saturation
                    score += qf * (tf / (tf + 1.0 + 0.5 * doc_len / 100.0))

            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:_BM25_PREFILTER_THRESHOLD]]

    def _llm_select_files(self, manifest_text: str, query: str) -> list[str]:
        """Single LLM call to select relevant files from manifest."""
        try:
            chat = self._chat_factory("fast")
            prompt = (
                "You are a file selection assistant. Given a user query and a manifest "
                "of available files, select the most relevant files.\n\n"
                f"Query: {query}\n\n"
                f"Available files:\n{manifest_text}\n\n"
                "Instructions:\n"
                f"- Select up to {_LLM_MAX_FILES} files most likely to contain the answer\n"
                "- Prefer files with relevant symbols, headings, or paths\n"
                "- Return ONLY a JSON array of file paths: "
                '[\"path/to/file1.py\", \"path/to/file2.md\"]\n'
                "- If no files seem relevant, return an empty array: []"
            )
            response = chat.chat([{"role": "user", "content": prompt}])
            text = response.strip()

            # Parse JSON array from response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                if isinstance(parsed, list):
                    return [
                        str(p) for p in parsed[:_LLM_MAX_FILES]
                        if isinstance(p, str) and p.strip()
                    ]
        except Exception as e:
            logger.warning(f"Agentic file selection failed: {e}")

        return []

    def _read_from_disk(self, entry: "ManifestEntry") -> str | None:
        """Read file content directly from source directory."""
        try:
            path = Path(entry.abs_path)
            if not path.exists():
                # Try relative to source_dir
                path = self._source_dir / entry.rel_path
            if not path.exists():
                return None
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"Cannot read {entry.rel_path} from disk: {e}")
            return None

    def _create_addresses(
        self, entry: "ManifestEntry", content: str
    ) -> list[Address]:
        """Create Address objects from a manifest entry.

        For code files: one SYMBOL address per symbol from manifest.
        For doc files: one SECTION address per heading, plus one FILE address.
        For other files: one FILE address.
        """
        addresses: list[Address] = []

        if entry.symbols:
            # Code file — create symbol addresses with AST line ranges
            for sym in entry.symbols:
                addresses.append(
                    Address(
                        kind=AddressKind.SYMBOL,
                        source_id=entry.file_id,
                        location=f"{entry.rel_path}:{sym.start_line}",
                        summary=f"{sym.kind} {sym.qualified_name}",
                        score=0.5,  # Neutral score; LLM already selected this file
                        metadata={
                            "start_line": sym.start_line,
                            "end_line": sym.end_line,
                            "name": sym.name,
                            "kind": sym.kind,
                            "signature": sym.signature,
                            "disk_path": entry.rel_path,
                            "agentic": True,
                        },
                    )
                )
        elif entry.headings:
            # Doc file — one file-level address with heading metadata
            heading_list = ", ".join(h.title for h in entry.headings[:5])
            addresses.append(
                Address(
                    kind=AddressKind.FILE,
                    source_id=entry.file_id,
                    location=entry.rel_path,
                    summary=f"Document: {heading_list}",
                    score=0.5,
                    metadata={
                        "disk_path": entry.rel_path,
                        "agentic": True,
                    },
                )
            )
        else:
            # Generic file
            addresses.append(
                Address(
                    kind=AddressKind.FILE,
                    source_id=entry.file_id,
                    location=entry.rel_path,
                    summary=f"File: {entry.rel_path}",
                    score=0.5,
                    metadata={
                        "disk_path": entry.rel_path,
                        "agentic": True,
                    },
                )
            )

        return addresses


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re

    return [t.lower() for t in re.split(r"[\s/._\-:;,(){}[\]]+", text) if len(t) > 1]


def _entry_to_text(entry: "ManifestEntry") -> str:
    """Build searchable text from a manifest entry for BM25."""
    parts = [entry.rel_path]
    for sym in entry.symbols:
        parts.append(sym.name)
        parts.append(sym.qualified_name)
        if sym.signature:
            parts.append(sym.signature)
    for heading in entry.headings:
        parts.append(heading.title)
    return " ".join(parts)
