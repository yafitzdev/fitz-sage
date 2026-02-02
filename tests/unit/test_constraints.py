# tests/test_constraints.py
"""
Tests for the Constraint Plugin system.

These tests verify:
1. ConflictAwareConstraint detects contradictions using LLM
2. Resolution queries bypass conflict detection
3. Pipeline integration works correctly

Uses mock chat provider for conflict detection and mock embedder for semantic matching.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

# Pure logic tests - run on every commit
pytestmark = pytest.mark.tier1

from fitz_ai.core.chunk import Chunk
from fitz_ai.core.governance import AnswerGovernor
from fitz_ai.core.guardrails import (
    ConflictAwareConstraint,
    ConstraintResult,
    SemanticMatcher,
    run_constraints,
)

from .mock_embedder import create_deterministic_embedder

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


def create_mock_chat(has_conflict: bool = False, conflicts: list[dict] | None = None):
    """Create a mock chat provider for testing conflict detection."""
    mock = MagicMock()
    if conflicts is None:
        conflicts = []
    response = json.dumps({"has_conflict": has_conflict, "conflicts": conflicts})
    mock.chat.return_value = response
    return mock


@pytest.fixture
def semantic_matcher() -> SemanticMatcher:
    """Create a semantic matcher with mock embedder for testing."""
    embedder = create_deterministic_embedder()
    return SemanticMatcher(
        embedder=embedder,
        causal_threshold=0.70,
        assertion_threshold=0.70,
        query_threshold=0.70,
    )


@pytest.fixture
def mock_chat_no_conflict():
    """Mock chat that returns no conflicts."""
    return create_mock_chat(has_conflict=False)


@pytest.fixture
def mock_chat_with_conflict():
    """Mock chat that returns a conflict."""
    return create_mock_chat(
        has_conflict=True,
        conflicts=[{"chunk_a": "1", "chunk_b": "2", "description": "conflicting claims"}],
    )


# =============================================================================
# Tests: ConstraintResult
# =============================================================================


class TestConstraintResult:
    """Test ConstraintResult factory methods."""

    def test_allow_factory(self):
        result = ConstraintResult.allow()
        assert result.allow_decisive_answer is True
        assert result.reason is None

    def test_deny_factory(self):
        result = ConstraintResult.deny("test reason", key="value")
        assert result.allow_decisive_answer is False
        assert result.reason == "test reason"
        assert result.metadata["key"] == "value"


# =============================================================================
# Tests: ConflictAwareConstraint
# =============================================================================


class TestConflictAwareConstraint:
    """Test the default conflict detection constraint.

    Note: ConflictAwareConstraint now uses LLM-based conflict detection.
    Tests use mock chat providers to simulate LLM responses.
    """

    def test_no_conflict_allows_answer(self, semantic_matcher, mock_chat_no_conflict):
        """Should allow decisive answer when no conflicts."""
        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat_no_conflict
        )

        chunks = [
            make_chunk("1", "Incident 17B was a security incident."),
            make_chunk("2", "The security incident caused significant downtime."),
        ]

        result = constraint.apply("Was Incident 17B a security incident?", chunks)

        assert result.allow_decisive_answer is True

    def test_detects_conflict_via_llm(self, semantic_matcher, mock_chat_with_conflict):
        """Should detect conflicts when LLM identifies them."""
        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat_with_conflict
        )

        chunks = [
            make_chunk("1", "Incident 17B was a security incident with unauthorized access."),
            make_chunk(
                "2", "Incident 17B was an operational incident with misconfigured settings."
            ),
        ]

        result = constraint.apply("Was Incident 17B a security incident?", chunks)

        assert result.allow_decisive_answer is False
        assert result.signal == "disputed"

    def test_empty_chunks_allows_answer(self, semantic_matcher, mock_chat_with_conflict):
        """Should allow when no chunks retrieved."""
        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat_with_conflict
        )

        result = constraint.apply("Any question?", [])

        assert result.allow_decisive_answer is True

    def test_single_chunk_allows_answer(self, semantic_matcher, mock_chat_with_conflict):
        """Should allow when only one chunk (can't have conflict)."""
        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat_with_conflict
        )

        chunks = [make_chunk("1", "This is a security incident.")]
        result = constraint.apply("What type?", chunks)

        assert result.allow_decisive_answer is True

    def test_disabled_constraint_always_allows(self, semantic_matcher, mock_chat_with_conflict):
        """Should allow when constraint is disabled."""
        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat_with_conflict, enabled=False
        )

        chunks = [
            make_chunk("1", "This is a security incident."),
            make_chunk("2", "This is an operational incident."),
        ]

        result = constraint.apply("What type?", chunks)

        assert result.allow_decisive_answer is True

    def test_resolution_query_allows_despite_conflict(
        self, semantic_matcher, mock_chat_with_conflict
    ):
        """Should allow decisive answer when query asks for resolution."""
        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat_with_conflict
        )

        chunks = [
            make_chunk("1", "Source A says: security incident."),
            make_chunk("2", "Source B says: operational incident."),
        ]

        # Explicitly asking for resolution - this bypasses conflict detection
        result = constraint.apply(
            "Which classification should be considered authoritative, and why?",
            chunks,
        )

        assert result.allow_decisive_answer is True

    def test_no_chat_skips_conflict_detection(self, semantic_matcher):
        """Should skip conflict detection when no chat provider available."""
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher, chat=None)

        chunks = [
            make_chunk("1", "This is a security incident."),
            make_chunk("2", "This is an operational incident."),
        ]

        # Without chat, conflict detection is skipped
        result = constraint.apply("What type?", chunks)

        assert result.allow_decisive_answer is True

    def test_llm_error_gracefully_allows(self, semantic_matcher):
        """Should gracefully handle LLM errors and allow."""
        mock_chat = MagicMock()
        mock_chat.chat.side_effect = Exception("LLM error")

        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat
        )

        chunks = [
            make_chunk("1", "Positive feedback."),
            make_chunk("2", "Negative feedback."),
        ]

        result = constraint.apply("What was the feedback?", chunks)

        # Errors should not block the answer
        assert result.allow_decisive_answer is True

    def test_malformed_llm_response_gracefully_allows(self, semantic_matcher):
        """Should gracefully handle malformed LLM responses."""
        mock_chat = MagicMock()
        mock_chat.chat.return_value = "not valid json at all"

        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat
        )

        chunks = [
            make_chunk("1", "Success."),
            make_chunk("2", "Failure."),
        ]

        result = constraint.apply("What happened?", chunks)

        # Malformed response should not block
        assert result.allow_decisive_answer is True

    def test_extracts_json_from_markdown_code_block(self, semantic_matcher):
        """Should extract JSON from markdown code blocks."""
        mock_chat = MagicMock()
        mock_chat.chat.return_value = """Here's the analysis:
```json
{"has_conflict": true, "conflicts": [{"chunk_a": "1", "chunk_b": "2", "description": "test"}]}
```
"""
        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat
        )

        chunks = [
            make_chunk("1", "A says yes."),
            make_chunk("2", "B says no."),
        ]

        result = constraint.apply("What?", chunks)
        assert result.allow_decisive_answer is False

    def test_extracts_json_with_surrounding_text(self, semantic_matcher):
        """Should extract JSON even with text before/after."""
        mock_chat = MagicMock()
        mock_chat.chat.return_value = """Based on analysis:
{"has_conflict": true, "conflicts": [{"chunk_a": "1", "chunk_b": "2", "description": "conflict"}]}
That's my finding."""

        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat
        )

        chunks = [
            make_chunk("1", "X is true."),
            make_chunk("2", "X is false."),
        ]

        result = constraint.apply("Is X true?", chunks)
        assert result.allow_decisive_answer is False


# =============================================================================
# Tests: Constraint Runner
# =============================================================================


class TestRunConstraints:
    """Test the constraint runner."""

    def test_no_constraints_returns_empty(self):
        """Should return empty list when no constraints configured."""
        results = run_constraints("query", [], [])
        assert results == []

    def test_single_constraint_deny(self, semantic_matcher, mock_chat_with_conflict):
        """Should preserve denial signal when single constraint denies."""
        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat_with_conflict
        )

        chunks = [
            make_chunk("1", "This is a security incident with unauthorized access."),
            make_chunk("2", "This is an operational incident from misconfiguration."),
        ]

        results = run_constraints("What type?", chunks, [constraint])
        governor = AnswerGovernor()
        decision = governor.decide(results)

        assert decision.mode.value in ("disputed", "qualified", "abstain")
        assert not decision.is_confident

    def test_multiple_constraints_any_deny(self):
        """Should preserve all signals when multiple constraints deny."""

        class AlwaysAllowConstraint:
            name = "always_allow"

            def apply(self, query, chunks):
                return ConstraintResult.allow()

        class AlwaysDenyConstraint:
            name = "always_deny"

            def apply(self, query, chunks):
                return ConstraintResult.deny("test denial")

        constraints = [AlwaysAllowConstraint(), AlwaysDenyConstraint()]

        results = run_constraints("query", [], constraints)
        governor = AnswerGovernor()
        decision = governor.decide(results)

        assert not decision.is_confident
        assert "test denial" in decision.reasons

    def test_constraint_exception_continues(self):
        """Should continue if a constraint raises an exception."""

        class CrashingConstraint:
            name = "crasher"

            def apply(self, query, chunks):
                raise RuntimeError("Oops!")

        class WorkingConstraint:
            name = "working"

            def apply(self, query, chunks):
                return ConstraintResult.allow()

        constraints = [CrashingConstraint(), WorkingConstraint()]

        # Should not raise, should allow (fail-safe)
        results = run_constraints("query", [], constraints)
        governor = AnswerGovernor()
        decision = governor.decide(results)

        assert decision.is_confident


# =============================================================================
# Tests: Acceptance Criteria (from handoff)
# =============================================================================


class TestAcceptanceCriteria:
    """
    Tests matching the acceptance criteria from the handoff document.

    These are the key behavioral tests that must pass.
    """

    def test_contradiction_query_must_surface_disagreement(
        self, semantic_matcher, mock_chat_with_conflict
    ):
        """
        Acceptance test: Contradiction detection.

        Query: "Was Incident 17B a security incident?"

        With conflicting docs, the constraint MUST deny decisive answer.
        The system should then surface disagreement (handled by generation).
        """
        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat_with_conflict
        )

        chunks = [
            make_chunk(
                "doc_a",
                "Incident 17B Investigation Report: This was a security incident "
                "involving unauthorized access attempts.",
            ),
            make_chunk(
                "doc_b",
                "Incident 17B Post-Mortem: This was an operational incident "
                "with misconfigured firewall settings.",
            ),
        ]

        result = constraint.apply("Was Incident 17B a security incident?", chunks)

        # MUST deny decisive answer when conflicts exist
        assert result.allow_decisive_answer is False
        assert result.reason is not None

    def test_authority_resolution_query_allowed(
        self, semantic_matcher, mock_chat_with_conflict
    ):
        """
        Acceptance test: Authority resolution.

        Query: "Which classification should be considered authoritative, and why?"

        Even with conflicting docs, this query explicitly asks for resolution,
        so the constraint should allow a decisive answer.
        """
        # Note: resolution query bypasses LLM conflict check entirely
        constraint = ConflictAwareConstraint(
            semantic_matcher=semantic_matcher, chat=mock_chat_with_conflict
        )

        chunks = [
            make_chunk(
                "doc_a",
                "Security team classified Incident 17B as a security incident.",
            ),
            make_chunk(
                "doc_b",
                "Operations team logged Incident 17B as an operational incident.",
            ),
        ]

        result = constraint.apply(
            "Which classification should be considered authoritative, and why?",
            chunks,
        )

        # MUST allow decisive answer for resolution queries
        assert result.allow_decisive_answer is True
