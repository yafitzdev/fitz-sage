# tests/e2e_krag/test_retrieval_e2e.py
"""
End-to-end tests for KRAG retrieval intelligence features.

Equivalent to tests/e2e/test_retrieval_e2e.py but runs through FitzKragEngine
instead of RAGPipeline. Uses the same test scenarios.

Tiered Execution (default):
- All scenarios run through tiered fallback (local -> cloud) during fixture setup
- Individual tests just look up their pre-computed result

Usage:
    # Run all KRAG E2E tests (tiered mode - default)
    pytest tests/e2e_krag/ -v -m e2e_krag

    # Run with single tier only (faster, local only)
    E2E_SINGLE_TIER=1 pytest tests/e2e_krag/ -v -m e2e_krag

    # Run specific feature tests
    pytest tests/e2e_krag/ -v -k "multi_hop"
"""

from __future__ import annotations

import pytest

from tests.e2e.scenarios import SCENARIOS

# Mark all tests in this module as e2e_krag tests
pytestmark = pytest.mark.e2e_krag


@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=lambda s: f"{s.id}_{s.feature.value}",
)
def test_scenario(krag_e2e_runner, scenario):
    """
    Run a single E2E scenario through KRAG engine.

    In tiered mode (default): looks up pre-computed result from tiered execution.
    In single-tier mode: runs scenario directly with first tier.
    """
    tiered_result = krag_e2e_runner.get_tiered_result(scenario.id)

    if tiered_result is not None:
        result, tier = tiered_result
    else:
        result = krag_e2e_runner.run_scenario(scenario)
        tier = krag_e2e_runner._current_tier or "unknown"

    if not result.validation.passed:
        msg = (
            f"\n\nScenario {scenario.id} ({scenario.name}) FAILED\n"
            f"Feature: {scenario.feature.value}\n"
            f"Tier: {tier}\n"
            f"Query: {scenario.query}\n"
            f"Reason: {result.validation.reason}\n"
            f"Details: {result.validation.details}\n"
            f"Answer preview: {result.answer_text[:300]}..."
        )
        pytest.fail(msg)


def test_full_suite_with_report(krag_e2e_runner):
    """
    Print summary report of tiered execution.

    In tiered mode: Uses pre-computed results from fixture setup.
    In single-tier mode: Runs all scenarios and generates report.
    """
    if hasattr(krag_e2e_runner, "_tiered_results") and krag_e2e_runner._tiered_results is not None:
        tiered_result = krag_e2e_runner._tiered_results
        tiered_result.print_summary()

        if tiered_result.pass_rate < 50:
            pytest.fail(
                f"Overall pass rate too low: {tiered_result.pass_rate:.1f}% "
                f"({tiered_result.total_passed}/{tiered_result.total})"
            )
    else:
        result = krag_e2e_runner.run_all()

        if result.pass_rate < 50:
            pytest.fail(
                f"Overall pass rate too low: {result.pass_rate:.1f}% "
                f"({result.passed}/{result.total})"
            )
