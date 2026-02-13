# fitz_ai/engines/fitz_krag/progressive/builder.py
"""
ManifestBuilder — fast directory scan with AST symbol + heading extraction.

No LLM calls, no embedding, no PostgreSQL. Runs in <500ms for 100 files.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from fitz_ai.engines.fitz_krag.progressive.manifest import (
    FileManifest,
    FileState,
    ManifestEntry,
    ManifestHeading,
    ManifestSymbol,
)

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig

logger = logging.getLogger(__name__)

# Extensions we track (code + docs + tables + rich docs)
_CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go"}
_DOC_EXTENSIONS = {".md", ".rst", ".txt"}
_TABLE_EXTENSIONS = {".csv", ".xlsx"}
_RICH_DOC_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm"}
_ALL_EXTENSIONS = _CODE_EXTENSIONS | _DOC_EXTENSIONS | _TABLE_EXTENSIONS | _RICH_DOC_EXTENSIONS

# Directories to skip
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".tox", ".eggs"}

# Heading regex for markdown
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Max bytes to probe for binary detection
_BINARY_PROBE_SIZE = 8192


def _is_text_file(path: Path) -> bool:
    """Check if an extensionless file is text by probing for null bytes."""
    try:
        chunk = path.read_bytes()[:_BINARY_PROBE_SIZE]
        return b"\x00" not in chunk and len(chunk) > 0
    except Exception:
        return False


class ManifestBuilder:
    """Builds a FileManifest from a source directory using fast extraction."""

    def __init__(self, config: "FitzKragConfig") -> None:
        self._config = config

    def build(self, source: Path, manifest_path: Path) -> FileManifest:
        """Scan directory, extract symbols/headings, create manifest.

        Reuses:
        - PythonCodeIngestStrategy().extract() for .py (stdlib ast, ~50ms/file)
        - _extract_headings() for .md/.rst/.txt (regex, instant)

        No LLM calls, no embedding calls, no PostgreSQL.
        """
        manifest = FileManifest(manifest_path)
        existing = manifest.entries()

        # Clear old entries — each point() replaces the manifest completely
        manifest.clear()

        # When source is a single file, use its parent as the base directory
        base_dir = source.parent if source.is_file() else source

        file_paths = self._scan_files(source)
        for abs_path in file_paths:
            rel_path = self._relative_path(abs_path, base_dir)
            ext = abs_path.suffix.lower()

            # Rich docs (PDF, DOCX, etc.) — hash raw bytes, no text extraction
            if ext in _RICH_DOC_EXTENSIONS:
                try:
                    raw_bytes = abs_path.read_bytes()
                except Exception as e:
                    logger.warning(f"Cannot read {abs_path}: {e}")
                    continue

                content_hash = hashlib.sha256(raw_bytes).hexdigest()
                existing_entry = existing.get(rel_path)
                if existing_entry and existing_entry.content_hash == content_hash:
                    manifest.add(existing_entry)
                    continue

                file_id = existing_entry.file_id if existing_entry else str(uuid.uuid4())
                entry = ManifestEntry(
                    file_id=file_id,
                    rel_path=rel_path,
                    abs_path=str(abs_path),
                    content_hash=content_hash,
                    file_type=ext,
                    size_bytes=len(raw_bytes),
                    state=FileState.REGISTERED,
                    symbols=[],
                    headings=[],
                )
                manifest.add(entry)
                continue

            # Text-based files — read as text, extract symbols/headings
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Cannot read {abs_path}: {e}")
                continue

            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Re-add unchanged files (manifest was cleared, so we must re-add)
            existing_entry = existing.get(rel_path)
            if existing_entry and existing_entry.content_hash == content_hash:
                manifest.add(existing_entry)
                continue

            file_id = existing_entry.file_id if existing_entry else str(uuid.uuid4())

            # Extract symbols or headings
            symbols: list[ManifestSymbol] = []
            headings: list[ManifestHeading] = []

            if ext == ".py":
                symbols = self._extract_python_symbols(content, rel_path)
            elif ext in {".ts", ".tsx", ".js", ".jsx"}:
                symbols = self._extract_ts_symbols(content, rel_path)
            elif ext == ".java":
                symbols = self._extract_java_symbols(content, rel_path)
            elif ext == ".go":
                symbols = self._extract_go_symbols(content, rel_path)
            elif ext in _DOC_EXTENSIONS or ext == "":
                headings = self._extract_headings(content)

            entry = ManifestEntry(
                file_id=file_id,
                rel_path=rel_path,
                abs_path=str(abs_path),
                content_hash=content_hash,
                file_type=ext,
                size_bytes=len(content.encode()),
                state=FileState.REGISTERED,
                symbols=symbols,
                headings=headings,
            )
            manifest.add(entry)

        manifest.save()
        return manifest

    def _scan_files(self, source: Path) -> list[Path]:
        """Scan directory for trackable files."""
        if source.is_file():
            if source.suffix.lower() in _ALL_EXTENSIONS or _is_text_file(source):
                return [source]
            return []

        files: list[Path] = []
        for ext in _ALL_EXTENSIONS:
            files.extend(source.rglob(f"*{ext}"))

        # Also pick up extensionless text files
        for f in source.rglob("*"):
            if f.is_file() and not f.suffix and _is_text_file(f):
                files.append(f)

        return sorted(
            f for f in files
            if not any(part in _SKIP_DIRS for part in f.parts)
        )

    def _relative_path(self, abs_path: Path, source: Path) -> str:
        """Get relative path string with forward slashes."""
        try:
            return str(abs_path.relative_to(source)).replace("\\", "/")
        except ValueError:
            return str(abs_path).replace("\\", "/")

    def _extract_python_symbols(
        self, content: str, file_path: str
    ) -> list[ManifestSymbol]:
        """Extract symbols from Python source using PythonCodeIngestStrategy."""
        try:
            from fitz_ai.engines.fitz_krag.ingestion.strategies.python_code import (
                PythonCodeIngestStrategy,
            )

            strategy = PythonCodeIngestStrategy()
            result = strategy.extract(content, file_path)

            return [
                ManifestSymbol(
                    name=sym.name,
                    qualified_name=sym.qualified_name,
                    kind=sym.kind,
                    signature=sym.signature,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                )
                for sym in result.symbols
            ]
        except Exception as e:
            logger.debug(f"Python symbol extraction failed for {file_path}: {e}")
            return []

    def _extract_ts_symbols(
        self, content: str, file_path: str
    ) -> list[ManifestSymbol]:
        """Extract symbols from TypeScript/JavaScript using TypeScriptIngestStrategy."""
        try:
            from fitz_ai.engines.fitz_krag.ingestion.strategies.typescript import (
                TypeScriptIngestStrategy,
            )

            strategy = TypeScriptIngestStrategy()
            result = strategy.extract(content, file_path)

            return [
                ManifestSymbol(
                    name=sym.name,
                    qualified_name=sym.qualified_name,
                    kind=sym.kind,
                    signature=sym.signature,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                )
                for sym in result.symbols
            ]
        except Exception as e:
            logger.debug(f"TypeScript symbol extraction failed for {file_path}: {e}")
            return []

    def _extract_java_symbols(
        self, content: str, file_path: str
    ) -> list[ManifestSymbol]:
        """Extract symbols from Java source using JavaIngestStrategy."""
        try:
            from fitz_ai.engines.fitz_krag.ingestion.strategies.java import (
                JavaIngestStrategy,
            )

            strategy = JavaIngestStrategy()
            result = strategy.extract(content, file_path)

            return [
                ManifestSymbol(
                    name=sym.name,
                    qualified_name=sym.qualified_name,
                    kind=sym.kind,
                    signature=sym.signature,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                )
                for sym in result.symbols
            ]
        except Exception as e:
            logger.debug(f"Java symbol extraction failed for {file_path}: {e}")
            return []

    def _extract_go_symbols(
        self, content: str, file_path: str
    ) -> list[ManifestSymbol]:
        """Extract symbols from Go source using GoIngestStrategy."""
        try:
            from fitz_ai.engines.fitz_krag.ingestion.strategies.go import (
                GoIngestStrategy,
            )

            strategy = GoIngestStrategy()
            result = strategy.extract(content, file_path)

            return [
                ManifestSymbol(
                    name=sym.name,
                    qualified_name=sym.qualified_name,
                    kind=sym.kind,
                    signature=sym.signature,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                )
                for sym in result.symbols
            ]
        except Exception as e:
            logger.debug(f"Go symbol extraction failed for {file_path}: {e}")
            return []

    def _extract_headings(self, content: str) -> list[ManifestHeading]:
        """Extract headings from markdown/rst/text files using regex."""
        headings: list[ManifestHeading] = []
        for match in _MD_HEADING_RE.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append(ManifestHeading(title=title, level=level))
        return headings
