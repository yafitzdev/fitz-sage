# tests/e2e/test_docling_parser.py
"""
E2E parser tests for Docling PDF and DOCX parsing.

These tests verify that the Docling parser correctly parses PDF and DOCX
documents into structured elements that can be queried by the RAG pipeline.

Run separately from main e2e tests:
    pytest -m e2e_parser

Run main e2e tests (excludes parser tests):
    pytest -m "e2e and not e2e_parser"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Mark all tests in this module as e2e_parser
pytestmark = pytest.mark.e2e_parser

# Fixtures directory (separate from main e2e fixtures to avoid slow ingestion)
FIXTURES_DIR = Path(__file__).parent / "fixtures_parser"


# =============================================================================
# Parser Test Result Types
# =============================================================================


@dataclass
class ParserTestResult:
    """Result of a single parser test."""

    test_name: str
    document_type: str  # "PDF" or "DOCX"
    passed: bool
    duration_ms: float
    error: str | None = None


@dataclass
class ParserRunResult:
    """Result of running all parser tests."""

    results: list[ParserTestResult] = field(default_factory=list)
    total_duration_s: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    def by_document_type(self) -> dict[str, dict[str, int]]:
        """Group results by document type."""
        stats: dict[str, dict[str, int]] = {}
        for r in self.results:
            if r.document_type not in stats:
                stats[r.document_type] = {"passed": 0, "failed": 0, "total": 0}
            stats[r.document_type]["total"] += 1
            if r.passed:
                stats[r.document_type]["passed"] += 1
            else:
                stats[r.document_type]["failed"] += 1
        return stats

    def print_summary(self) -> None:
        """Print a summary report."""
        print("\n" + "=" * 60)
        print("DOCLING PARSER E2E TEST RESULTS")
        print("=" * 60)

        by_type = self.by_document_type()
        for doc_type, stats in sorted(by_type.items()):
            status = "OK" if stats["failed"] == 0 else "!!"
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"  [{status}] {doc_type:<15} {stats['passed']}/{stats['total']} ({rate:.0f}%)")

        print("-" * 60)
        print(f"  Total: {self.passed}/{self.total} passed ({self.pass_rate:.1f}%)")
        print(f"  Duration: {self.total_duration_s:.1f}s")
        print("=" * 60)


# Global storage for test results
_parser_results: ParserRunResult | None = None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def parser_router():
    """Create a shared ParserRouter for all tests."""
    from fitz_ai.ingestion.parser.router import ParserRouter

    return ParserRouter()


@pytest.fixture(scope="module")
def sample_pdf() -> Path:
    """Return path to sample PDF fixture."""
    return FIXTURES_DIR / "sample.pdf"


@pytest.fixture(scope="module")
def sample_docx() -> Path:
    """Return path to sample DOCX fixture."""
    return FIXTURES_DIR / "sample.docx"


@pytest.fixture(scope="module")
def parsed_pdf(parser_router, sample_pdf):
    """Parse the sample PDF once and cache the result."""
    from fitz_ai.ingestion.source.base import SourceFile

    source = SourceFile(
        uri=f"file://{sample_pdf}",
        local_path=sample_pdf,
        size=sample_pdf.stat().st_size,
    )
    return parser_router.parse(source)


@pytest.fixture(scope="module")
def parsed_docx(parser_router, sample_docx):
    """Parse the sample DOCX once and cache the result."""
    from fitz_ai.ingestion.source.base import SourceFile

    source = SourceFile(
        uri=f"file://{sample_docx}",
        local_path=sample_docx,
        size=sample_docx.stat().st_size,
    )
    return parser_router.parse(source)


@pytest.fixture(scope="module")
def parser_run_results(parsed_pdf, parsed_docx):
    """
    Run all parser validation checks and store results.

    This fixture runs once per module and caches all results for individual tests.
    """
    global _parser_results

    print("\n" + "=" * 60)
    print("DOCLING PARSER E2E TESTS")
    print("=" * 60 + "\n")

    results = ParserRunResult()
    start_time = time.time()

    # Define all test checks
    checks = [
        # PDF checks
        ("pdf_parses_successfully", "PDF", lambda: _check_parses(parsed_pdf)),
        ("pdf_extracts_headings", "PDF", lambda: _check_pdf_headings(parsed_pdf)),
        ("pdf_extracts_tables", "PDF", lambda: _check_pdf_tables(parsed_pdf)),
        ("pdf_extracts_text_content", "PDF", lambda: _check_pdf_text(parsed_pdf)),
        # DOCX checks
        ("docx_parses_successfully", "DOCX", lambda: _check_parses(parsed_docx)),
        ("docx_extracts_headings", "DOCX", lambda: _check_docx_headings(parsed_docx)),
        ("docx_extracts_tables", "DOCX", lambda: _check_docx_tables(parsed_docx)),
        ("docx_extracts_text_content", "DOCX", lambda: _check_docx_text(parsed_docx)),
        # Integration checks
        ("both_formats_consistent", "Integration", lambda: _check_consistency(parsed_pdf, parsed_docx)),
    ]

    print("--- Running parser checks ---\n")

    for test_name, doc_type, check_fn in checks:
        check_start = time.time()
        try:
            check_fn()
            passed = True
            error = None
            status = "PASS"
        except AssertionError as e:
            passed = False
            error = str(e)
            status = "FAIL"
        except Exception as e:
            passed = False
            error = f"Error: {e}"
            status = "FAIL"

        duration_ms = (time.time() - check_start) * 1000
        results.results.append(
            ParserTestResult(
                test_name=test_name,
                document_type=doc_type,
                passed=passed,
                duration_ms=duration_ms,
                error=error,
            )
        )
        print(f"  [{status}] {doc_type}/{test_name} ({duration_ms:.0f}ms)")

    results.total_duration_s = time.time() - start_time

    # Print summary
    results.print_summary()

    _parser_results = results
    return results


# =============================================================================
# Check Functions (called by fixture)
# =============================================================================


def _check_parses(result: Any) -> None:
    """Check that document parsed successfully."""
    assert result is not None
    assert len(result.elements) > 0


def _check_pdf_headings(result: Any) -> None:
    """Check that PDF extracts headings."""
    headings = [e for e in result.elements if e.type.name == "HEADING"]
    assert len(headings) >= 1
    assert any("Nexus Robotics" in h.content for h in headings)


def _check_pdf_tables(result: Any) -> None:
    """Check that PDF extracts tables."""
    tables = [e for e in result.elements if e.type.name == "TABLE"]
    assert len(tables) >= 1
    table_content = " ".join(t.content for t in tables)
    assert "500 kg" in table_content or "Maximum Payload" in table_content


def _check_pdf_text(result: Any) -> None:
    """Check that PDF extracts text content."""
    all_content = " ".join(e.content for e in result.elements)
    assert "Sarah Chen" in all_content
    assert "847" in all_content
    assert "124.5" in all_content or "124.5 million" in all_content


def _check_docx_headings(result: Any) -> None:
    """Check that DOCX extracts headings."""
    headings = [e for e in result.elements if e.type.name == "HEADING"]
    assert len(headings) >= 1
    assert any("CloudScale" in h.content for h in headings)


def _check_docx_tables(result: Any) -> None:
    """Check that DOCX extracts tables."""
    tables = [e for e in result.elements if e.type.name == "TABLE"]
    assert len(tables) >= 1
    table_content = " ".join(t.content for t in tables)
    assert "Kafka" in table_content or "Redis" in table_content


def _check_docx_text(result: Any) -> None:
    """Check that DOCX extracts text content."""
    all_content = " ".join(e.content for e in result.elements)
    assert "DataFlow" in all_content
    assert "Seattle" in all_content
    assert "2.3 petabytes" in all_content or "2.3 PB" in all_content


def _check_consistency(pdf_result: Any, docx_result: Any) -> None:
    """Check that both formats produce consistent element types."""
    pdf_types = {e.type.name for e in pdf_result.elements}
    docx_types = {e.type.name for e in docx_result.elements}

    assert "HEADING" in pdf_types
    assert "HEADING" in docx_types
    assert "TEXT" in pdf_types
    assert "TEXT" in docx_types


# =============================================================================
# Helper to Get Pre-computed Results
# =============================================================================


def _get_result(test_name: str) -> ParserTestResult | None:
    """Get pre-computed result for a test."""
    global _parser_results
    if _parser_results is None:
        return None
    for r in _parser_results.results:
        if r.test_name == test_name:
            return r
    return None


# =============================================================================
# Test Classes (Use Pre-computed Results)
# =============================================================================


class TestDoclingParserPDF:
    """Tests for Docling PDF parsing."""

    def test_pdf_parses_successfully(self, parser_run_results):
        """PDF should parse without errors."""
        result = _get_result("pdf_parses_successfully")
        assert result is not None
        if not result.passed:
            pytest.fail(f"PDF parse failed: {result.error}")

    def test_pdf_extracts_headings(self, parser_run_results):
        """PDF should extract heading elements."""
        result = _get_result("pdf_extracts_headings")
        assert result is not None
        if not result.passed:
            pytest.fail(f"PDF heading extraction failed: {result.error}")

    def test_pdf_extracts_tables(self, parser_run_results):
        """PDF should extract table elements."""
        result = _get_result("pdf_extracts_tables")
        assert result is not None
        if not result.passed:
            pytest.fail(f"PDF table extraction failed: {result.error}")

    def test_pdf_extracts_text_content(self, parser_run_results):
        """PDF should extract queryable text content."""
        result = _get_result("pdf_extracts_text_content")
        assert result is not None
        if not result.passed:
            pytest.fail(f"PDF text extraction failed: {result.error}")


class TestDoclingParserDOCX:
    """Tests for Docling DOCX parsing."""

    def test_docx_parses_successfully(self, parser_run_results):
        """DOCX should parse without errors."""
        result = _get_result("docx_parses_successfully")
        assert result is not None
        if not result.passed:
            pytest.fail(f"DOCX parse failed: {result.error}")

    def test_docx_extracts_headings(self, parser_run_results):
        """DOCX should extract heading elements."""
        result = _get_result("docx_extracts_headings")
        assert result is not None
        if not result.passed:
            pytest.fail(f"DOCX heading extraction failed: {result.error}")

    def test_docx_extracts_tables(self, parser_run_results):
        """DOCX should extract table elements."""
        result = _get_result("docx_extracts_tables")
        assert result is not None
        if not result.passed:
            pytest.fail(f"DOCX table extraction failed: {result.error}")

    def test_docx_extracts_text_content(self, parser_run_results):
        """DOCX should extract queryable text content."""
        result = _get_result("docx_extracts_text_content")
        assert result is not None
        if not result.passed:
            pytest.fail(f"DOCX text extraction failed: {result.error}")


class TestDoclingParserIntegration:
    """Integration tests for parser with multiple document types."""

    def test_both_formats_parse_consistently(self, parser_run_results):
        """Both PDF and DOCX should produce structured elements."""
        result = _get_result("both_formats_consistent")
        assert result is not None
        if not result.passed:
            pytest.fail(f"Consistency check failed: {result.error}")

    def test_full_suite_summary(self, parser_run_results):
        """Print final summary (always passes if we got here)."""
        # Summary already printed by fixture, this test just ensures it ran
        assert parser_run_results.total > 0
        if parser_run_results.pass_rate < 50:
            pytest.fail(
                f"Overall pass rate too low: {parser_run_results.pass_rate:.1f}% "
                f"({parser_run_results.passed}/{parser_run_results.total})"
            )
