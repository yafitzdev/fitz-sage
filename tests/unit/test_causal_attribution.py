# tests/unit/test_causal_attribution.py
"""
Tests for CausalAttributionConstraint.

These tests verify:
1. Causal queries without causal evidence are denied
2. Causal queries with explicit causal language are allowed
3. Non-causal queries pass through
4. Without embedder the constraint is a no-op (provider presence = feature toggle)

Detection is semantic (embedding-based). Tests use a keyword-aware mock embedder
that maps concept categories to orthogonal unit vectors, making cosine similarity
binary and deterministic.
"""

from __future__ import annotations

import hashlib

import pytest

# Pure logic tests - run on every commit
pytestmark = pytest.mark.tier1

from fitz_ai.core.chunk import Chunk
from fitz_ai.governance import CausalAttributionConstraint
from fitz_ai.governance.constraints.semantic import (
    CAUSAL_CONCEPTS,
    CAUSAL_QUERY_CONCEPTS,
    HEDGE_EVIDENCE_CONCEPTS,
    OPINION_QUERY_CONCEPTS,
    PREDICTIVE_EVIDENCE_CONCEPTS,
    PREDICTIVE_QUERY_CONCEPTS,
    SPECULATIVE_QUERY_CONCEPTS,
)

# =============================================================================
# Mock Embedder
# =============================================================================


# Direct lookup for SemanticMatcher anchor phrases → canonical category bucket.
# This avoids false bucket collisions when an anchor phrase happens to contain
# a keyword from a different category (e.g. CAUSAL_QUERY_CONCEPTS includes
# "what will happen next" which keyword-matches "predictive_query").
_ANCHOR_CATEGORY: dict[str, str] = {}
for _phrase in CAUSAL_CONCEPTS:
    _ANCHOR_CATEGORY[_phrase] = "causal_evidence"
for _phrase in CAUSAL_QUERY_CONCEPTS:
    _ANCHOR_CATEGORY[_phrase] = "causal_query"
for _phrase in PREDICTIVE_QUERY_CONCEPTS:
    _ANCHOR_CATEGORY[_phrase] = "predictive_query"
for _phrase in PREDICTIVE_EVIDENCE_CONCEPTS:
    _ANCHOR_CATEGORY[_phrase] = "predictive_evidence"
for _phrase in OPINION_QUERY_CONCEPTS:
    _ANCHOR_CATEGORY[_phrase] = "opinion_query"
for _phrase in SPECULATIVE_QUERY_CONCEPTS:
    _ANCHOR_CATEGORY[_phrase] = "speculative_query"
for _phrase in HEDGE_EVIDENCE_CONCEPTS:
    _ANCHOR_CATEGORY[_phrase] = "hedge_evidence"

# Keyword-based classification for test input texts (queries and chunks).
# Only these inputs need keyword detection; anchor phrases are handled above.
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "causal_query": ("why", "what caused", "what led", "explain why", "how come", "what made"),
    "causal_evidence": (
        "because",
        "due to",
        "caused by",
        "led to",
        "as a result",
        "therefore",
        "thus",
        "consequently",
        "reason",
        "attributed to",
    ),
    "predictive_query": (
        "will ",
        "what will",
        "how will",
        "forecast",
        "predict",
        "projection",
        "outlook",
        "anticipated",
        "next year",
        "next quarter",
        "next week",
        "next month",
        "going to",
        "be like in",
        "estimate for",
    ),
    "predictive_evidence": (
        "projected",
        "expected to",
        "anticipated to",
        "will likely",
        "estimated to reach",
    ),
    "opinion_query": (
        "should we",
        "should i",
        "is it better",
        "which is better",
        "which is best",
        "is it worth",
        "recommend",
        " better for ",
        " better than ",
        "what's the best",
        "what is the best",
        "is best",
    ),
    "speculative_query": (
        "will succeed",
        "will fail",
        "will be approved",
        "will become",
        "be successful",
        "be mainstream",
        "what percentage will",
        "how many will",
    ),
}

_DIM = 32  # Vector dimensionality for mock


def _category_vector(category: str) -> list[float]:
    """Return a unit vector for a concept category (deterministic via hash)."""
    idx = int(hashlib.md5(category.encode()).hexdigest(), 16) % _DIM
    vec = [0.0] * _DIM
    vec[idx] = 1.0
    return vec


def _mock_embed(text: str) -> list[float]:
    """Embed text into a category vector.

    Anchor phrases from SemanticMatcher are resolved via a direct lookup table
    so they always land in the correct concept group regardless of keyword
    collisions. All other texts (test queries and chunks) fall through to
    keyword-based detection.
    """
    if text in _ANCHOR_CATEGORY:
        return _category_vector(_ANCHOR_CATEGORY[text])
    text_lower = text.lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return _category_vector(cat)
    return _category_vector("general")


class MockEmbedder:
    """Keyword-aware mock embedder for unit tests."""

    def embed(self, text: str, task_type: str = "query") -> list[float]:
        return _mock_embed(text)


def make_constraint() -> CausalAttributionConstraint:
    return CausalAttributionConstraint(embedder=MockEmbedder())


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
        """Non-causal queries should pass through."""
        constraint = make_constraint()

        chunks = [make_chunk("1", "Helios was deprecated because of high costs.")]

        result = constraint.apply("What is Helios?", chunks)

        assert result.allow_decisive_answer is True

    def test_disabled_always_allows(self):
        """Should allow when disabled."""
        constraint = CausalAttributionConstraint(enabled=False, embedder=MockEmbedder())

        chunks = [make_chunk("1", "Helios was deprecated.")]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.allow_decisive_answer is True

    def test_empty_chunks_defers(self):
        """Empty chunks should defer to other constraints."""
        constraint = make_constraint()

        result = constraint.apply("Why did X happen?", [])

        assert result.allow_decisive_answer is True

    def test_no_embedder_is_noop(self):
        """Without embedder the constraint is a no-op — provider presence = feature toggle."""
        constraint = CausalAttributionConstraint()

        chunks = [make_chunk("1", "Helios was deprecated.")]

        # Even a clear causal query with no evidence should pass through
        result = constraint.apply("Why was Helios deprecated?", chunks)

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
        """Should detect causal queries and deny when no causal evidence exists."""
        constraint = make_constraint()

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
        """Non-causal queries should pass through."""
        constraint = make_constraint()

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
        constraint = make_constraint()

        chunks = [make_chunk("1", f"Helios was deprecated {causal_phrase}.")]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.allow_decisive_answer is True

    def test_denies_without_causal_markers(self):
        """Should deny when no causal language is present."""
        constraint = make_constraint()

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
        constraint = make_constraint()

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
        constraint = make_constraint()

        chunks = [
            make_chunk("1", "Helios was deprecated in 2023."),
            make_chunk("2", "This was due to high operational costs."),  # Has causal
            make_chunk("3", "Orion is now primary."),
        ]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.allow_decisive_answer is True

    def test_metadata_includes_chunk_counts(self):
        """Denial should include chunk count metadata."""
        constraint = make_constraint()

        chunks = [
            make_chunk("1", "System A was deprecated."),
            make_chunk("2", "System B replaced it."),
        ]

        result = constraint.apply("Why was System A deprecated?", chunks)

        assert result.metadata.get("total_chunks") == 2


# =============================================================================
# Tests: Semantic Detection Properties
# =============================================================================


class TestSemanticDetectionProperties:
    """
    Test semantic-based implementation properties.

    Detection uses SemanticMatcher (embedding similarity). With the mock embedder,
    results are deterministic — same input always produces same output.
    """

    def test_requires_embedder_for_detection(self):
        """CausalAttributionConstraint requires embedder for detection."""
        constraint = CausalAttributionConstraint()
        assert constraint.enabled is True
        assert constraint._semantic_matcher is None

    def test_deterministic_results(self):
        """Same input should produce same result (mock embedder is deterministic)."""
        constraint = make_constraint()
        chunks = [make_chunk("1", "Helios was deprecated.")]
        query = "Why was Helios deprecated?"

        results = [constraint.apply(query, chunks) for _ in range(3)]

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
        """Should detect predictive queries and deny when no predictive evidence exists."""
        constraint = make_constraint()

        # No predictive evidence
        chunks = [make_chunk("1", "The system was launched in 2023.")]

        result = constraint.apply(query, chunks)

        assert result.allow_decisive_answer is False

    def test_allows_with_predictive_evidence(self):
        """Should allow predictive queries when forecast/projection evidence exists."""
        constraint = make_constraint()

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
        """Should detect opinion queries and deny (no definitive evidence for opinions)."""
        constraint = make_constraint()

        # Factual content without recommendations
        chunks = [make_chunk("1", "PostgreSQL is a relational database.")]

        result = constraint.apply(query, chunks)

        assert result.allow_decisive_answer is False
        assert result.signal == "qualified"
