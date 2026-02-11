# tests/e2e_krag/test_pdf_docx_retrieval.py
"""
KRAG E2E retrieval tests for PDF and DOCX documents.

Equivalent to tests/e2e/test_pdf_docx_retrieval.py but uses FitzKragEngine.
Requires the PDF/DOCX fixtures in fixtures_parser/ directory.

Run with: pytest -m e2e_krag_parser

Tiered Execution:
- All scenarios run through tiered fallback (local -> cloud) during fixture setup
- Individual tests just look up their pre-computed result
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e_krag.scenarios import PDF_DOCX_SCENARIOS

from .conftest import create_tiered_krag_runner

# Mark all tests in this module as e2e_krag_parser (slow)
pytestmark = pytest.mark.e2e_krag_parser

# Fixtures directory with PDF/DOCX files (shared with RAG e2e)
FIXTURES_PARSER_DIR = Path(__file__).parent / "fixtures_parser"


@pytest.fixture(scope="module")
def krag_pdf_docx_runner(set_workspace):
    """
    Module-scoped KRAG runner for PDF/DOCX retrieval tests.

    Uses shared tiered runner factory from conftest.py.
    """
    yield from create_tiered_krag_runner(
        FIXTURES_PARSER_DIR, PDF_DOCX_SCENARIOS, "KRAG PDF/DOCX E2E"
    )()


@pytest.mark.parametrize(
    "scenario",
    PDF_DOCX_SCENARIOS,
    ids=lambda s: f"{s.id}_{s.feature.value}",
)
def test_pdf_docx_scenario(krag_pdf_docx_runner, scenario):
    """
    Run a PDF/DOCX retrieval scenario through KRAG engine.

    In tiered mode (default): looks up pre-computed result from tiered execution.
    In single-tier mode: runs scenario directly with first tier.
    """
    tiered_result = krag_pdf_docx_runner.get_tiered_result(scenario.id)

    if tiered_result is not None:
        result, tier = tiered_result
    else:
        result = krag_pdf_docx_runner.run_scenario(scenario)
        tier = krag_pdf_docx_runner._current_tier or "unknown"

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
