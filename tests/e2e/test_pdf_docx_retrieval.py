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

import os
from pathlib import Path

import pytest

from .runner import E2ERunner, TieredRunResult
from .scenarios import PDF_DOCX_SCENARIOS

# Mark all tests in this module as e2e_parser (slow)
pytestmark = pytest.mark.e2e_parser

# Fixtures directory with PDF/DOCX files
FIXTURES_PARSER_DIR = Path(__file__).parent / "fixtures_parser"

# Global storage for tiered results (populated once per module)
_pdf_docx_tiered_results: TieredRunResult | None = None


@pytest.fixture(scope="module")
def pdf_docx_runner():
    """
    Module-scoped runner for PDF/DOCX retrieval tests.

    Runs ALL scenarios through tiered execution (local -> cloud) during setup.
    Individual tests then look up their pre-computed result.
    """
    global _pdf_docx_tiered_results

    runner = E2ERunner(fixtures_dir=FIXTURES_PARSER_DIR, use_cache=True)
    runner.setup()

    # Run tiered execution unless disabled
    use_tiered = os.environ.get("E2E_SINGLE_TIER", "0") != "1"

    if use_tiered:
        print("\n" + "=" * 60)
        print("PDF/DOCX E2E TESTS - TIERED EXECUTION")
        print("(local -> cloud fallback)")
        print("Set E2E_SINGLE_TIER=1 to disable")
        print("=" * 60 + "\n")

        _pdf_docx_tiered_results = runner.run_tiered(PDF_DOCX_SCENARIOS)
        runner._tiered_results = _pdf_docx_tiered_results
    else:
        _pdf_docx_tiered_results = None
        runner._tiered_results = None

    yield runner
    runner.teardown()


def _get_tiered_result(scenario_id: str):
    """Get pre-computed result for a scenario from tiered execution."""
    global _pdf_docx_tiered_results
    if _pdf_docx_tiered_results is None:
        return None
    return _pdf_docx_tiered_results.results.get(scenario_id)


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
    tiered_result = _get_tiered_result(scenario.id)

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
