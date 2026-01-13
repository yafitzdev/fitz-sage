# tests/e2e/test_retrieval_e2e.py
"""
End-to-end tests for retrieval intelligence features.

These tests validate that all retrieval intelligence features work correctly
with real document ingestion and RAG pipeline execution.

Usage:
    # Run all E2E tests
    pytest tests/e2e/ -v -m e2e

    # Run specific feature tests
    pytest tests/e2e/ -v -k "multi_hop"

    # Run with full output
    pytest tests/e2e/ -v -s -m e2e
"""

from __future__ import annotations

import pytest

from .reporter import E2EReporter
from .scenarios import SCENARIOS, Feature, get_scenarios_by_feature

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

    Each scenario tests a specific retrieval intelligence feature.
    """
    result = e2e_runner.run_scenario(scenario)

    # Provide detailed failure message
    if not result.validation.passed:
        msg = (
            f"\n\nScenario {scenario.id} ({scenario.name}) FAILED\n"
            f"Feature: {scenario.feature.value}\n"
            f"Query: {scenario.query}\n"
            f"Reason: {result.validation.reason}\n"
            f"Details: {result.validation.details}\n"
            f"Answer preview: {result.answer_text[:300]}..."
        )
        pytest.fail(msg)


# =============================================================================
# Feature-Specific Test Classes
# =============================================================================


class TestMultiHop:
    """Tests for multi-hop retrieval."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.MULTI_HOP),
        ids=lambda s: s.id,
    )
    def test_multi_hop_scenario(self, e2e_runner, scenario):
        """Test multi-hop reasoning chains."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestEntityGraph:
    """Tests for entity graph expansion."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.ENTITY_GRAPH),
        ids=lambda s: s.id,
    )
    def test_entity_graph_scenario(self, e2e_runner, scenario):
        """Test entity-based retrieval expansion."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestComparison:
    """Tests for comparison query detection."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.COMPARISON),
        ids=lambda s: s.id,
    )
    def test_comparison_scenario(self, e2e_runner, scenario):
        """Test comparison query handling."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestConflictAware:
    """Tests for conflict detection."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.CONFLICT_AWARE),
        ids=lambda s: s.id,
    )
    def test_conflict_scenario(self, e2e_runner, scenario):
        """Test conflict detection and hedged responses."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestInsufficientEvidence:
    """Tests for insufficient evidence handling."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.INSUFFICIENT_EVIDENCE),
        ids=lambda s: s.id,
    )
    def test_insufficient_evidence_scenario(self, e2e_runner, scenario):
        """Test abstaining when evidence is missing."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestCausalAttribution:
    """Tests for causal attribution detection."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.CAUSAL_ATTRIBUTION),
        ids=lambda s: s.id,
    )
    def test_causal_scenario(self, e2e_runner, scenario):
        """Test causal claim handling."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestTableQueries:
    """Tests for tabular data queries."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.TABLE_SCHEMA)
        + get_scenarios_by_feature(Feature.TABLE_QUERY),
        ids=lambda s: s.id,
    )
    def test_table_scenario(self, e2e_runner, scenario):
        """Test CSV/table query handling."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestCodeSearch:
    """Tests for code search and understanding."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.CODE_SEARCH),
        ids=lambda s: s.id,
    )
    def test_code_scenario(self, e2e_runner, scenario):
        """Test code content retrieval."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestLongDocument:
    """Tests for long document retrieval."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.LONG_DOC),
        ids=lambda s: s.id,
    )
    def test_long_doc_scenario(self, e2e_runner, scenario):
        """Test retrieval from long documents."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestBasicRetrieval:
    """Tests for basic retrieval sanity checks."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.BASIC_RETRIEVAL),
        ids=lambda s: s.id,
    )
    def test_basic_scenario(self, e2e_runner, scenario):
        """Test basic fact retrieval."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestFreshness:
    """Tests for freshness/authority boosting."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.FRESHNESS),
        ids=lambda s: s.id,
    )
    def test_freshness_scenario(self, e2e_runner, scenario):
        """Test authority boosting with official/spec keywords."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestHybridSearch:
    """Tests for hybrid search (dense + sparse)."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.HYBRID_SEARCH),
        ids=lambda s: s.id,
    )
    def test_hybrid_search_scenario(self, e2e_runner, scenario):
        """Test hybrid search with exact keyword matching."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestQueryExpansion:
    """Tests for query expansion (synonym/acronym variations)."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.QUERY_EXPANSION),
        ids=lambda s: s.id,
    )
    def test_query_expansion_scenario(self, e2e_runner, scenario):
        """Test query expansion with synonym and acronym variations."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestTemporal:
    """Tests for temporal query handling (time-based comparisons and periods)."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.TEMPORAL),
        ids=lambda s: s.id,
    )
    def test_temporal_scenario(self, e2e_runner, scenario):
        """Test temporal query handling with time references."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


class TestAggregation:
    """Tests for aggregation query handling (list all, count, enumerate)."""

    @pytest.mark.parametrize(
        "scenario",
        get_scenarios_by_feature(Feature.AGGREGATION),
        ids=lambda s: s.id,
    )
    def test_aggregation_scenario(self, e2e_runner, scenario):
        """Test aggregation query handling with comprehensive retrieval."""
        result = e2e_runner.run_scenario(scenario)
        assert result.validation.passed, result.validation.reason


# =============================================================================
# Full Suite Test with Report
# =============================================================================


def test_full_suite_with_report(e2e_runner):
    """
    Run all scenarios and generate a report.

    This test runs all scenarios and prints a summary report.
    Individual scenario failures are collected but don't fail this test.
    """
    result = e2e_runner.run_all()

    # Generate and print report
    reporter = E2EReporter(result)
    reporter.console_report()

    # This test passes if majority (>50%) of tests pass
    # Individual tests above handle strict pass/fail
    if result.pass_rate < 50:
        pytest.fail(
            f"Overall pass rate too low: {result.pass_rate:.1f}% "
            f"({result.passed}/{result.total})"
        )
