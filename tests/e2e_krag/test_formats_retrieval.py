# tests/e2e_krag/test_formats_retrieval.py
"""
KRAG E2E retrieval tests for additional file formats.

Tests PPTX, XLSX, SQL, Go, Java, and TypeScript file parsing and retrieval.
Requires the fixtures in fixtures_formats/ directory.

Run with: pytest -m e2e_krag_formats
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e_krag.scenarios import FORMAT_SCENARIOS

from .conftest import create_tiered_krag_runner

# Mark all tests in this module as e2e_krag_formats
pytestmark = pytest.mark.e2e_krag_formats

FIXTURES_FORMATS_DIR = Path(__file__).parent / "fixtures_formats"


@pytest.fixture(scope="module")
def krag_formats_runner(set_workspace):
    """
    Module-scoped KRAG runner for format-specific retrieval tests.

    Uses shared tiered runner factory from conftest.py.
    """
    yield from create_tiered_krag_runner(
        FIXTURES_FORMATS_DIR, FORMAT_SCENARIOS, "KRAG FORMATS E2E"
    )()


@pytest.mark.parametrize(
    "scenario",
    FORMAT_SCENARIOS,
    ids=lambda s: f"{s.id}_{s.feature.value}",
)
def test_formats_scenario(krag_formats_runner, scenario):
    """
    Run a format-specific retrieval scenario through KRAG engine.

    In tiered mode (default): looks up pre-computed result from tiered execution.
    In single-tier mode: runs scenario directly with first tier.
    """
    tiered_result = krag_formats_runner.get_tiered_result(scenario.id)

    if tiered_result is not None:
        result, tier = tiered_result
    else:
        result = krag_formats_runner.run_scenario(scenario)
        tier = krag_formats_runner._current_tier or "unknown"

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
