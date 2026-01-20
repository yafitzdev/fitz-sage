# tests/e2e/test_docling_parser.py
"""
E2E tests for Docling PDF and DOCX parsing.

Tests that the Docling parser correctly parses PDF and DOCX files.
Logs detailed timing information to .e2e_cache/docling_timing.log for debugging.

Note: First PDF parse takes ~70s due to TensorFlow/RapidOCR model loading.
Subsequent parses are fast (~3s) as the model stays loaded in memory.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fitz_ai.core.document import ParsedDocument

# Log file for timing information (in .e2e_cache for consistency)
CACHE_DIR = Path(__file__).parent / ".e2e_cache"
LOG_FILE = CACHE_DIR / "docling_timing.log"


def log(message: str) -> None:
    """Log message with timestamp to both console and file."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def timed(name: str):
    """Context manager for timing code blocks."""

    class Timer:
        def __init__(self, name: str):
            self.name = name
            self.start = None
            self.elapsed = None

        def __enter__(self):
            log(f"START: {self.name}")
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
            log(f"END: {self.name} ({self.elapsed:.2f}s)")

    return Timer(name)


@pytest.fixture(scope="module")
def log_setup():
    """Initialize log file at start of module."""
    # Ensure cache directory exists
    CACHE_DIR.mkdir(exist_ok=True)
    # Clear log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== Docling Parser Test Run: {datetime.now().isoformat()} ===\n\n")
    yield LOG_FILE
    log(f"\nLog file written to: {LOG_FILE}")


# Static fixture files in tests/e2e/fixtures/
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def sample_pdf(log_setup) -> Path:
    """Return path to sample PDF fixture file."""
    pdf_path = FIXTURES_DIR / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip(f"PDF fixture not found: {pdf_path}")
    log(f"Using PDF fixture: {pdf_path} ({pdf_path.stat().st_size} bytes)")
    return pdf_path


@pytest.fixture(scope="module")
def sample_docx(log_setup) -> Path:
    """Return path to sample DOCX fixture file."""
    docx_path = FIXTURES_DIR / "sample.docx"
    if not docx_path.exists():
        pytest.skip(f"DOCX fixture not found: {docx_path}")
    log(f"Using DOCX fixture: {docx_path} ({docx_path.stat().st_size} bytes)")
    return docx_path


@pytest.mark.e2e
@pytest.mark.slow
class TestDoclingParser:
    """E2E tests for Docling parser with detailed timing."""

    def test_import_docling(self, log_setup):
        """Test that docling can be imported and measure import time."""
        with timed("Import docling module"):
            import docling  # noqa: F401

        log("Docling import successful")

    def test_import_parser_router(self, log_setup):
        """Test importing the parser router (which loads docling)."""
        with timed("Import ParserRouter"):
            from fitz_ai.ingestion.parser.router import ParserRouter  # noqa: F401

        log("ParserRouter import successful")

    def test_create_parser_router(self, log_setup):
        """Test creating a parser router instance."""
        with timed("Import ParserRouter"):
            from fitz_ai.ingestion.parser.router import ParserRouter

        with timed("Create ParserRouter instance"):
            router = ParserRouter()

        log(f"Router created with {len(router._parsers)} registered parsers")

    def test_parse_pdf(self, sample_pdf: Path, log_setup):
        """Test parsing a PDF file with detailed timing."""
        from fitz_ai.ingestion.parser.router import ParserRouter
        from fitz_ai.ingestion.source.base import SourceFile

        with timed("Create ParserRouter"):
            router = ParserRouter()

        with timed("Create SourceFile"):
            source = SourceFile(
                uri=f"file://{sample_pdf}",
                local_path=sample_pdf,
                size=sample_pdf.stat().st_size,
            )

        with timed("Parse PDF with Docling"):
            result: ParsedDocument = router.parse(source)

        log(f"Parsed PDF: {len(result.elements)} elements")
        for i, elem in enumerate(result.elements[:5]):
            log(f"  Element {i}: {elem.type.name} - {elem.content[:50]!r}...")

        assert result is not None
        assert len(result.elements) > 0

    def test_parse_docx(self, sample_docx: Path, log_setup):
        """Test parsing a DOCX file with detailed timing."""
        from fitz_ai.ingestion.parser.router import ParserRouter
        from fitz_ai.ingestion.source.base import SourceFile

        with timed("Create ParserRouter"):
            router = ParserRouter()

        with timed("Create SourceFile"):
            source = SourceFile(
                uri=f"file://{sample_docx}",
                local_path=sample_docx,
                size=sample_docx.stat().st_size,
            )

        with timed("Parse DOCX with Docling"):
            result: ParsedDocument = router.parse(source)

        log(f"Parsed DOCX: {len(result.elements)} elements")
        for i, elem in enumerate(result.elements[:5]):
            log(f"  Element {i}: {elem.type.name} - {elem.content[:50]!r}...")

        assert result is not None
        assert len(result.elements) > 0

    def test_parse_both_sequential(self, sample_pdf: Path, sample_docx: Path, log_setup):
        """Test parsing both PDF and DOCX sequentially with timing."""
        from fitz_ai.ingestion.parser.router import ParserRouter
        from fitz_ai.ingestion.source.base import SourceFile

        log("\n--- Sequential Parse Test ---")

        with timed("Total sequential parse time"):
            with timed("Create ParserRouter (shared)"):
                router = ParserRouter()

            # Parse PDF
            with timed("Parse PDF"):
                pdf_source = SourceFile(
                    uri=f"file://{sample_pdf}",
                    local_path=sample_pdf,
                    size=sample_pdf.stat().st_size,
                )
                pdf_result = router.parse(pdf_source)

            # Parse DOCX
            with timed("Parse DOCX"):
                docx_source = SourceFile(
                    uri=f"file://{sample_docx}",
                    local_path=sample_docx,
                    size=sample_docx.stat().st_size,
                )
                docx_result = router.parse(docx_source)

        log(f"PDF elements: {len(pdf_result.elements)}")
        log(f"DOCX elements: {len(docx_result.elements)}")

        assert len(pdf_result.elements) > 0
        assert len(docx_result.elements) > 0
