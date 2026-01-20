# tests/e2e/test_retrieval_e2e.py
"""
End-to-end tests for retrieval intelligence features.

These tests validate that all retrieval intelligence features work correctly
with real document ingestion and RAG pipeline execution.

Tiered Execution (default):
- All scenarios run through tiered fallback (local -> cloud) during fixture setup
- Individual tests just look up their pre-computed result
- This saves time and tokens by only using cloud for failures

Usage:
    # Run all E2E tests (tiered mode - default)
    pytest tests/e2e/ -v -m e2e

    # Run with single tier only (faster, local only)
    E2E_SINGLE_TIER=1 pytest tests/e2e/ -v -m e2e

    # Run specific feature tests
    pytest tests/e2e/ -v -k "multi_hop"

    # Run with full output
    pytest tests/e2e/ -v -s -m e2e
"""

from __future__ import annotations

import pytest

from .reporter import E2EReporter
from .scenarios import SCENARIOS

# Mark all tests in this module as e2e tests
pytestmark = pytest.mark.e2e


# =============================================================================
# Parametrized Test for All Scenarios
# =============================================================================


@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=lambda s: f"{s.id}_{s.feature.value}",
)
def test_scenario(e2e_runner, scenario):
    """
    Run a single E2E scenario.

    In tiered mode (default): looks up pre-computed result from tiered execution.
    In single-tier mode: runs scenario directly with first tier.
    """
    # Check if we have pre-computed tiered results
    tiered_result = e2e_runner.get_tiered_result(scenario.id)

    if tiered_result is not None:
        # Use pre-computed result from tiered execution
        result, tier = tiered_result
    else:
        # Single-tier mode: run scenario directly
        result = e2e_runner.run_scenario(scenario)
        tier = e2e_runner._current_tier or "unknown"

    # Provide detailed failure message
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


# =============================================================================
# Full Suite Test with Report
# =============================================================================


def test_full_suite_with_report(e2e_runner):
    """
    Print summary report of tiered execution.

    In tiered mode: Uses pre-computed results from fixture setup.
    In single-tier mode: Runs all scenarios and generates report.
    """
    # Check if we have tiered results
    if hasattr(e2e_runner, "_tiered_results") and e2e_runner._tiered_results is not None:
        # Tiered mode: results already computed, just print summary again
        tiered_result = e2e_runner._tiered_results
        tiered_result.print_summary()

        # This test passes if majority (>50%) of tests pass
        if tiered_result.pass_rate < 50:
            pytest.fail(
                f"Overall pass rate too low: {tiered_result.pass_rate:.1f}% "
                f"({tiered_result.total_passed}/{tiered_result.total})"
            )
    else:
        # Single-tier mode: run all scenarios
        result = e2e_runner.run_all()

        # Generate and print report
        reporter = E2EReporter(result)
        reporter.console_report()

        # This test passes if majority (>50%) of tests pass
        if result.pass_rate < 50:
            pytest.fail(
                f"Overall pass rate too low: {result.pass_rate:.1f}% "
                f"({result.passed}/{result.total})"
            )
