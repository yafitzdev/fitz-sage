# tests/unit/test_causal_attribution.py
"""
Tests for CausalAttributionConstraint.

These tests verify:
1. Causal queries without causal evidence are denied
2. Causal queries with explicit causal language are allowed
3. Non-causal queries pass through

The constraint uses keyword-based detection (no embeddings, no LLM).
"""

from __future__ import annotations

import pytest

# Pure logic tests - run on every commit
pytestmark = pytest.mark.tier1

from fitz_ai.core.chunk import Chunk
from fitz_ai.core.guardrails import CausalAttributionConstraint

# =============================================================================
# Test Data
# =============================================================================


def make_chunk(id: str, content: str) -> Chunk:
    """Create a test chunk."""
    return Chunk(
        id=id,
        doc_id=f"doc_{id}",
        content=content,
        chunk_index=0,
        metadata={},
    )


# =============================================================================
# Tests: Basic Behavior
# =============================================================================


class TestBasicBehavior:
    """Test basic constraint behavior."""

    def test_non_causal_query_with_causal_evidence_allowed(self):
        """Non-causal queries should pass through (keyword detection)."""
        constraint = CausalAttributionConstraint()

        # Non-causal query (no "why", "what caused", etc.)
        chunks = [make_chunk("1", "Helios was deprecated because of high costs.")]

        result = constraint.apply("What is Helios?", chunks)

        assert result.allow_decisive_answer is True

    def test_disabled_always_allows(self):
        """Should allow when disabled."""
        constraint = CausalAttributionConstraint(enabled=False)

        chunks = [make_chunk("1", "Helios was deprecated.")]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.allow_decisive_answer is True

    def test_empty_chunks_defers(self):
        """Empty chunks should defer to other constraints."""
        constraint = CausalAttributionConstraint()

        result = constraint.apply("Why did X happen?", [])

        # Defers to InsufficientEvidenceConstraint
        assert result.allow_decisive_answer is True


# =============================================================================
# Tests: Causal Query Detection
# =============================================================================


class TestCausalQueryDetection:
    """Test detection of causal queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "Why was Helios deprecated?",
            "Why did the system fail?",
            "What caused the outage?",
            "What led to the incident?",
            "Explain why the system failed",
        ],
    )
    def test_detects_causal_queries(self, query: str):
        """Should detect causal queries via keywords."""
        constraint = CausalAttributionConstraint()

        # No causal evidence
        chunks = [make_chunk("1", "The system was deprecated.")]

        result = constraint.apply(query, chunks)

        assert result.allow_decisive_answer is False

    @pytest.mark.parametrize(
        "query",
        [
            "What is Helios?",
            "When was Helios deprecated?",
            "Who maintains the system?",
            "Where is the documentation?",
        ],
    )
    def test_non_causal_queries_pass(self, query: str):
        """Non-causal queries should pass through (no causal keywords)."""
        constraint = CausalAttributionConstraint()

        # No causal evidence needed for non-causal queries
        chunks = [make_chunk("1", "The system was deprecated.")]

        result = constraint.apply(query, chunks)

        assert result.allow_decisive_answer is True


# =============================================================================
# Tests: Causal Evidence Detection
# =============================================================================


class TestCausalEvidenceDetection:
    """Test detection of causal language in evidence."""

    @pytest.mark.parametrize(
        "causal_phrase",
        [
            "because of high costs",
            "due to reliability issues",
            "caused by a bug",
            "led to the outage",
            "as a result of the incident",
            "therefore we changed",
            "thus the system failed",
        ],
    )
    def test_allows_with_causal_markers(self, causal_phrase: str):
        """Should allow when causal language is present."""
        constraint = CausalAttributionConstraint()

        chunks = [make_chunk("1", f"Helios was deprecated {causal_phrase}.")]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.allow_decisive_answer is True

    def test_denies_without_causal_markers(self):
        """Should deny when no causal language is present."""
        constraint = CausalAttributionConstraint()

        chunks = [
            make_chunk("1", "Helios was deprecated in August 2023."),
            make_chunk("2", "The system is no longer maintained."),
            make_chunk("3", "Orion replaced Helios as primary system."),
        ]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.allow_decisive_answer is False
        assert "causal" in result.reason.lower()

    def test_signal_is_qualified_not_abstain(self):
        """Signal should be 'qualified' not 'abstain' (we have evidence, just not causal)."""
        constraint = CausalAttributionConstraint()

        chunks = [make_chunk("1", "Helios was deprecated.")]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.signal == "qualified"


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_causal_marker_in_any_chunk_allows(self):
        """Should allow if ANY chunk has causal language."""
        constraint = CausalAttributionConstraint()

        chunks = [
            make_chunk("1", "Helios was deprecated in 2023."),
            make_chunk("2", "This was due to high operational costs."),  # Has causal
            make_chunk("3", "Orion is now primary."),
        ]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.allow_decisive_answer is True

    def test_metadata_includes_chunk_counts(self):
        """Denial should include chunk count metadata."""
        constraint = CausalAttributionConstraint()

        chunks = [
            make_chunk("1", "System A was deprecated."),
            make_chunk("2", "System B replaced it."),
        ]

        result = constraint.apply("Why was System A deprecated?", chunks)

        assert result.metadata.get("total_chunks") == 2


# =============================================================================
# Tests: Keyword-Based Behavior
# =============================================================================


class TestKeywordBasedBehavior:
    """
    Test keyword-based implementation (no LLM, no embeddings).

    The CausalAttributionConstraint uses simple keyword matching for:
    - Detecting causal queries ("why", "what caused", etc.)
    - Detecting causal evidence ("because", "due to", etc.)
    """

    def test_no_external_dependencies(self):
        """CausalAttributionConstraint requires no external dependencies."""
        # Should work with no parameters - fully self-contained
        constraint = CausalAttributionConstraint()
        assert constraint.enabled is True

    def test_fast_execution_no_llm(self):
        """Constraint should execute quickly (keyword matching only)."""
        import time

        constraint = CausalAttributionConstraint()
        chunks = [
            make_chunk("1", "Helios was deprecated in August 2023."),
            make_chunk("2", "The system is no longer maintained."),
        ]

        start = time.perf_counter()
        _ = constraint.apply("Why was Helios deprecated?", chunks)  # noqa: F841
        elapsed = time.perf_counter() - start

        # Should be instant (< 100ms for keyword matching)
        assert elapsed < 0.1

    def test_deterministic_results(self):
        """Same input should produce same result (no LLM randomness)."""
        constraint = CausalAttributionConstraint()
        chunks = [make_chunk("1", "Helios was deprecated.")]
        query = "Why was Helios deprecated?"

        results = [constraint.apply(query, chunks) for _ in range(3)]

        # All results should be identical
        assert all(r.allow_decisive_answer == results[0].allow_decisive_answer for r in results)
        assert all(r.signal == results[0].signal for r in results)


# =============================================================================
# Tests: Predictive Query Detection
# =============================================================================


class TestPredictiveQueryDetection:
    """Test detection of predictive/future-oriented queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "Will the system be upgraded next year?",
            "What will happen to the project?",
            "How will this affect users?",
            "What is the forecast for adoption?",
            "What do you predict for Q3?",
        ],
    )
    def test_detects_predictive_queries(self, query: str):
        """Should detect predictive queries via keywords."""
        constraint = CausalAttributionConstraint()

        # No predictive evidence
        chunks = [make_chunk("1", "The system was launched in 2023.")]

        result = constraint.apply(query, chunks)

        assert result.allow_decisive_answer is False

    def test_allows_with_predictive_evidence(self):
        """Should allow predictive queries when forecast/projection evidence exists."""
        constraint = CausalAttributionConstraint()

        chunks = [
            make_chunk("1", "Adoption is projected to reach 50% by Q4."),
        ]

        result = constraint.apply("What will adoption look like next quarter?", chunks)

        assert result.allow_decisive_answer is True


# =============================================================================
# Tests: Opinion Query Detection
# =============================================================================


class TestOpinionQueryDetection:
    """Test detection of opinion/recommendation queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "Should we use this framework?",
            "Is it better to use Python or Go?",
            "Which database is best for this use case?",
            "Is it worth pursuing this approach?",
            "What do you recommend?",
        ],
    )
    def test_detects_opinion_queries(self, query: str):
        """Should detect opinion queries via keywords."""
        constraint = CausalAttributionConstraint()

        # Factual content without recommendations
        chunks = [make_chunk("1", "PostgreSQL is a relational database.")]

        result = constraint.apply(query, chunks)

        assert result.allow_decisive_answer is False
        assert result.signal == "qualified"
