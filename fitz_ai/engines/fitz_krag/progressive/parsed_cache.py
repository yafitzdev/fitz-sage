# fitz_ai/engines/fitz_krag/progressive/parsed_cache.py
"""
Parsed text cache for rich documents (PDF, DOCX, PPTX, HTML).

Caches Docling-parsed text by content hash so PDFs are only parsed once.
Cache lives at ~/.fitz/collections/{col}/parsed/{content_hash}.txt.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RICH_DOC_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm"}


def get_parsed_text(
    path: Path,
    content_hash: str,
    cache_dir: Path,
) -> str | None:
    """Get parsed text for a rich document, using cache when available.

    Args:
        path: Absolute path to the file on disk.
        content_hash: SHA-256 hash of the file's raw bytes.
        cache_dir: Directory to store/read cached parsed text.

    Returns:
        Parsed text, or None if parsing fails.
    """
    # Check cache first
    cache_path = cache_dir / f"{content_hash}.txt"
    if cache_path.exists():
        try:
            text = cache_path.read_text(encoding="utf-8")
            if text.strip():
                return text
        except Exception:
            pass  # Cache corrupt, re-parse

    # Parse with Docling
    text = _parse(path)
    if not text:
        return None

    # Save to cache
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    except Exception as e:
        logger.debug(f"Failed to cache parsed text for {path.name}: {e}")

    return text


_parser_logs_suppressed = False


def _parse(path: Path) -> str | None:
    """Parse a rich document using the parser system."""
    global _parser_logs_suppressed
    try:
        # Globally disable INFO logs during parsing — RapidOCR overrides
        # its logger level during init, so per-logger suppression doesn't work.
        import logging as _logging

        if not _parser_logs_suppressed:
            _logging.disable(_logging.INFO)

        from fitz_ai.ingestion.parser import ParserRouter
        from fitz_ai.ingestion.source.base import SourceFile

        source_file = SourceFile(uri=path.as_uri(), local_path=path)
        router = ParserRouter()
        parsed = router.parse(source_file)

        if not _parser_logs_suppressed:
            _logging.disable(_logging.NOTSET)
            _suppress_parser_logs()
            _parser_logs_suppressed = True

        text = parsed.full_text
        return text if text and text.strip() else None
    except Exception as e:
        # Re-enable logging even on failure
        if not _parser_logs_suppressed:
            import logging as _logging

            _logging.disable(_logging.NOTSET)
        logger.warning(f"Parser failed for {path.name}: {e}")
        return None


def _suppress_parser_logs() -> None:
    """Suppress noisy third-party logs from PDF/doc parsers permanently."""
    import logging as _logging

    for name in ("rapidocr", "docling", "RapidOCR",
                 "fitz_ai.ingestion.parser"):
        _logger = _logging.getLogger(name)
        _logger.setLevel(_logging.WARNING)
        _logger.handlers.clear()
