# tests/e2e/test_pdf_docx_retrieval.py
"""
E2E retrieval tests for PDF and DOCX documents.

These tests verify that PDF/DOCX content can be queried through the RAG pipeline.
Requires the PDF/DOCX fixtures in fixtures_parser/ directory.

Run with: pytest -m e2e_parser

Tiered Execution:
- All scenarios run through tiered fallback (local -> cloud) during fixture setup
- Individual tests just look up their pre-computed result
- Summary report printed after all tests complete
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import create_tiered_runner
from .scenarios import PDF_DOCX_SCENARIOS

# Mark all tests in this module as e2e_parser (slow)
pytestmark = pytest.mark.e2e_parser

# Fixtures directory with PDF/DOCX files
FIXTURES_PARSER_DIR = Path(__file__).parent / "fixtures_parser"


@pytest.fixture(scope="module")
def pdf_docx_runner(set_workspace):
    """
    Module-scoped runner for PDF/DOCX retrieval tests.

    Uses shared tiered runner factory from conftest.py.
    Runs ALL scenarios through tiered execution (local -> cloud) during setup.
    Individual tests then look up their pre-computed result via runner.get_tiered_result().
    """
    yield from create_tiered_runner(FIXTURES_PARSER_DIR, PDF_DOCX_SCENARIOS, "PDF/DOCX E2E")()


@pytest.mark.parametrize(
    "scenario",
    PDF_DOCX_SCENARIOS,
    ids=lambda s: f"{s.id}_{s.feature.value}",
)
def test_pdf_docx_scenario(pdf_docx_runner, scenario):
    """
    Run a PDF/DOCX retrieval scenario.

    In tiered mode (default): looks up pre-computed result from tiered execution.
    In single-tier mode: runs scenario directly with first tier.
    """
    # Check if we have pre-computed tiered results
    tiered_result = pdf_docx_runner.get_tiered_result(scenario.id)

    if tiered_result is not None:
        # Use pre-computed result from tiered execution
        result, tier = tiered_result
    else:
        # Single-tier mode: run scenario directly
        result = pdf_docx_runner.run_scenario(scenario)
        tier = pdf_docx_runner._current_tier or "unknown"

    if not result.validation.passed:
        msg = (
            f"\n\nScenario {scenario.id} ({scenario.name}) FAILED\n"
            f"Tier: {tier}\n"
            f"Feature: {scenario.feature.value}\n"
            f"Query: {scenario.query}\n"
            f"Reason: {result.validation.reason}\n"
            f"Details: {result.validation.details}\n"
            f"Answer preview: {result.answer_text[:300] if result.answer_text else '(no answer)'}..."
        )
        pytest.fail(msg)
