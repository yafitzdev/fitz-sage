# tests/test_constraints.py
"""
Tests for the Constraint Plugin system.

These tests verify:
1. ConflictAwareConstraint detects contradictions using simple YES/NO stance detection
2. Resolution queries bypass conflict detection
3. Pipeline integration works correctly

Uses mock chat provider for stance detection.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest

from fitz_ai.core.answer_mode import AnswerMode

# Pure logic tests - run on every commit
pytestmark = pytest.mark.tier1

from fitz_ai.core.chunk import Chunk
from fitz_ai.governance import (
    AnswerGovernor,
    ConflictAwareConstraint,
    ConstraintResult,
    run_constraints,
)
from fitz_ai.governance.constraints.semantic import RESOLUTION_QUERY_CONCEPTS

# =============================================================================
# Mock Embedder (for resolution query detection)
# =============================================================================

_DIM = 32


def _resolution_category_vector() -> list[float]:
    """Return the unit vector for the 'resolution_query' category."""
    idx = int(hashlib.md5(b"resolution_query").hexdigest(), 16) % _DIM
    vec = [0.0] * _DIM
    vec[idx] = 1.0
    return vec


# Anchor lookup: map each RESOLUTION_QUERY_CONCEPT to the resolution_query vector.
_RESOLUTION_ANCHORS: frozenset[str] = frozenset(RESOLUTION_QUERY_CONCEPTS)

# Keywords that flag a text as a resolution query.
_RESOLUTION_KEYWORDS = ("authoritative", "which source", "trust", "resolve", "reconcile")


def _resolution_mock_embed(text: str) -> list[float]:
    """Embed text: anchor phrases → resolution_query; keywords → resolution_query; else zeros."""
    if text in _RESOLUTION_ANCHORS or any(kw in text.lower() for kw in _RESOLUTION_KEYWORDS):
        return _resolution_category_vector()
    return [0.0] * _DIM


class MockEmbedder:
    """Minimal embedder that enables resolution query detection for ConflictAwareConstraint."""

    def embed(self, text: str, task_type: str = "query") -> list[float]:
        return _resolution_mock_embed(text)


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


def create_mock_chat_yes_no(stances: list[str]):
    """
    Create a mock chat provider that returns YES/NO stances.

    Args:
        stances: List of responses for each call ("YES", "NO", or "UNCLEAR")
    """
    mock = MagicMock()
    mock.chat.side_effect = stances
    return mock


def create_mock_chat_no_conflict():
    """Mock chat that returns AGREE (no contradiction) for pairwise checks."""
    mock = MagicMock()
    mock.chat.return_value = "AGREE"
    return mock


def create_mock_chat_with_conflict():
    """Mock chat that returns CONTRADICT for pairwise checks."""
    mock = MagicMock()
    mock.chat.return_value = "CONTRADICT"
    return mock


@pytest.fixture
def mock_chat_no_conflict():
    """Mock chat that returns no conflicts."""
    return create_mock_chat_no_conflict()


@pytest.fixture
def mock_chat_with_conflict():
    """Mock chat that returns a conflict."""
    return create_mock_chat_with_conflict()


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

    Note: ConflictAwareConstraint now uses simple YES/NO stance detection.
    Tests use mock chat providers to simulate LLM responses.
    """

    def test_no_conflict_allows_answer(self, mock_chat_no_conflict):
        """Should allow decisive answer when no conflicts."""
        constraint = ConflictAwareConstraint(chat=mock_chat_no_conflict)

        chunks = [
            make_chunk("1", "Incident 17B was a security incident."),
            make_chunk("2", "The security incident caused significant downtime."),
        ]

        result = constraint.apply("Was Incident 17B a security incident?", chunks)

        assert result.allow_decisive_answer is True

    def test_detects_conflict_via_stance(self, mock_chat_with_conflict):
        """Should detect conflicts when stances are YES vs NO."""
        constraint = ConflictAwareConstraint(chat=mock_chat_with_conflict)

        chunks = [
            make_chunk("1", "Incident 17B was a security incident with unauthorized access."),
            make_chunk(
                "2", "Incident 17B was an operational incident with misconfigured settings."
            ),
        ]

        result = constraint.apply("Was Incident 17B a security incident?", chunks)

        assert result.allow_decisive_answer is False
        assert result.signal == "disputed"

    def test_empty_chunks_allows_answer(self, mock_chat_with_conflict):
        """Should allow when no chunks retrieved."""
        constraint = ConflictAwareConstraint(chat=mock_chat_with_conflict)

        result = constraint.apply("Any question?", [])

        assert result.allow_decisive_answer is True

    def test_single_chunk_allows_answer(self, mock_chat_with_conflict):
        """Should allow when only one chunk (can't have conflict)."""
        constraint = ConflictAwareConstraint(chat=mock_chat_with_conflict)

        chunks = [make_chunk("1", "This is a security incident.")]
        result = constraint.apply("What type?", chunks)

        assert result.allow_decisive_answer is True

    def test_disabled_constraint_always_allows(self, mock_chat_with_conflict):
        """Should allow when constraint is disabled."""
        constraint = ConflictAwareConstraint(chat=mock_chat_with_conflict, enabled=False)

        chunks = [
            make_chunk("1", "This is a security incident."),
            make_chunk("2", "This is an operational incident."),
        ]

        result = constraint.apply("What type?", chunks)

        assert result.allow_decisive_answer is True

    def test_resolution_query_allows_despite_conflict(self, mock_chat_with_conflict):
        """Should allow decisive answer when query asks for resolution."""
        constraint = ConflictAwareConstraint(chat=mock_chat_with_conflict, embedder=MockEmbedder())

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

    def test_no_chat_skips_conflict_detection(self):
        """Should skip conflict detection when no chat provider available."""
        constraint = ConflictAwareConstraint(chat=None)

        chunks = [
            make_chunk("1", "This is a security incident."),
            make_chunk("2", "This is an operational incident."),
        ]

        # Without chat, conflict detection is skipped
        result = constraint.apply("What type?", chunks)

        assert result.allow_decisive_answer is True

    def test_llm_error_gracefully_allows(self):
        """Should gracefully handle LLM errors and allow."""
        mock_chat = MagicMock()
        mock_chat.chat.side_effect = Exception("LLM error")

        constraint = ConflictAwareConstraint(chat=mock_chat)

        chunks = [
            make_chunk("1", "Positive feedback."),
            make_chunk("2", "Negative feedback."),
        ]

        result = constraint.apply("What was the feedback?", chunks)

        # Errors should not block the answer (returns UNCLEAR for both)
        assert result.allow_decisive_answer is True

    def test_unclear_stances_allow_answer(self):
        """Should allow when all stances are UNCLEAR."""
        mock_chat = MagicMock()
        mock_chat.chat.return_value = "UNCLEAR"

        constraint = ConflictAwareConstraint(chat=mock_chat)

        chunks = [
            make_chunk("1", "Some tangential information."),
            make_chunk("2", "Other tangential information."),
        ]

        result = constraint.apply("What happened?", chunks)

        # UNCLEAR stances don't create conflict
        assert result.allow_decisive_answer is True

    def test_mixed_yes_unclear_allows_answer(self):
        """Should allow when stances are YES and UNCLEAR (no NO)."""
        mock_chat = MagicMock()
        mock_chat.chat.side_effect = ["YES", "UNCLEAR"]

        constraint = ConflictAwareConstraint(chat=mock_chat)

        chunks = [
            make_chunk("1", "Yes, it was approved."),
            make_chunk("2", "The project involved many stakeholders."),
        ]

        result = constraint.apply("Was it approved?", chunks)

        # YES + UNCLEAR is not a conflict
        assert result.allow_decisive_answer is True

    def test_mixed_no_unclear_allows_answer(self):
        """Should allow when stances are NO and UNCLEAR (no YES)."""
        mock_chat = MagicMock()
        mock_chat.chat.side_effect = ["NO", "UNCLEAR"]

        constraint = ConflictAwareConstraint(chat=mock_chat)

        chunks = [
            make_chunk("1", "The proposal was rejected."),
            make_chunk("2", "Budget discussions continued."),
        ]

        result = constraint.apply("Was it approved?", chunks)

        # NO + UNCLEAR is not a conflict
        assert result.allow_decisive_answer is True


# =============================================================================
# Tests: Constraint Runner
# =============================================================================


class TestRunConstraints:
    """Test the constraint runner."""

    def test_no_constraints_returns_empty(self):
        """Should return empty list when no constraints configured."""
        results = run_constraints("query", [], [])
        assert results == []

    def test_single_constraint_deny(self, mock_chat_with_conflict):
        """Should preserve denial signal when single constraint denies."""
        constraint = ConflictAwareConstraint(chat=mock_chat_with_conflict)

        chunks = [
            make_chunk("1", "This is a security incident with unauthorized access."),
            make_chunk("2", "This is an operational incident from misconfiguration."),
        ]

        results = run_constraints("What type?", chunks, [constraint])
        governor = AnswerGovernor()
        decision = governor.decide(results)

        assert decision.mode.value in ("disputed", "trustworthy", "abstain")

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

        # Denial without recognized signal resolves to TRUSTWORTHY
        # but the constraint is still tracked
        assert decision.mode == AnswerMode.TRUSTWORTHY
        assert "test denial" in decision.reasons
        assert len(decision.triggered_constraints) > 0

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

        assert decision.is_trustworthy


# =============================================================================
# Tests: Acceptance Criteria (from handoff)
# =============================================================================


class TestAcceptanceCriteria:
    """
    Tests matching the acceptance criteria from the handoff document.

    These are the key behavioral tests that must pass.
    """

    def test_contradiction_query_must_surface_disagreement(self, mock_chat_with_conflict):
        """
        Acceptance test: Contradiction detection.

        Query: "Was Incident 17B a security incident?"

        With conflicting docs, the constraint MUST deny decisive answer.
        The system should then surface disagreement (handled by generation).
        """
        constraint = ConflictAwareConstraint(chat=mock_chat_with_conflict)

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

    def test_authority_resolution_query_allowed(self, mock_chat_with_conflict):
        """
        Acceptance test: Authority resolution.

        Query: "Which classification should be considered authoritative, and why?"

        Even with conflicting docs, this query explicitly asks for resolution,
        so the constraint should allow a decisive answer.
        """
        # Resolution detection requires an embedder (SemanticMatcher); without
        # one the constraint cannot recognise resolution queries.
        constraint = ConflictAwareConstraint(chat=mock_chat_with_conflict, embedder=MockEmbedder())

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


# =============================================================================
# Tests: AnswerVerificationConstraint (Jury-based)
# =============================================================================


class TestAnswerVerificationConstraint:
    """Test the jury-based answer verification constraint.

    This constraint uses 3-prompt LLM jury to verify chunks actually answer
    the query. Requires 2+ NO votes to qualify (conservative).
    """

    def test_no_chat_allows_answer(self):
        """Should allow when no chat provider available."""
        from fitz_ai.governance import AnswerVerificationConstraint

        constraint = AnswerVerificationConstraint(chat=None)

        chunks = [make_chunk("1", "France has 67 million people.")]
        result = constraint.apply("What is the capital of France?", chunks)

        assert result.allow_decisive_answer is True

    def test_empty_chunks_allows_answer(self):
        """Should allow when no chunks (handled by InsufficientEvidence)."""
        from fitz_ai.governance import AnswerVerificationConstraint

        mock_chat = MagicMock()
        constraint = AnswerVerificationConstraint(chat=mock_chat)

        result = constraint.apply("Any question?", [])

        assert result.allow_decisive_answer is True
        mock_chat.chat.assert_not_called()

    def test_disabled_constraint_allows_answer(self):
        """Should allow when constraint is disabled."""
        from fitz_ai.governance import AnswerVerificationConstraint

        mock_chat = MagicMock()
        constraint = AnswerVerificationConstraint(chat=mock_chat, enabled=False)

        chunks = [make_chunk("1", "Some content.")]
        result = constraint.apply("Any question?", chunks)

        assert result.allow_decisive_answer is True
        mock_chat.chat.assert_not_called()

    def test_jury_unanimous_yes_allows_answer(self):
        """Should allow when all 3 jury prompts say context is relevant."""
        from fitz_ai.governance import AnswerVerificationConstraint

        mock_chat = MagicMock()
        # All 3 prompts: YES (context is relevant)
        mock_chat.chat.side_effect = ["YES", "YES", "YES"]
        constraint = AnswerVerificationConstraint(chat=mock_chat)

        chunks = [make_chunk("1", "Paris is the capital of France.")]
        result = constraint.apply("What is the capital of France?", chunks)

        assert result.allow_decisive_answer is True
        assert mock_chat.chat.call_count == 3

    def test_jury_2no_balanced_confirms_qualifies(self):
        """Should qualify when 2/3 fast say NO and balanced confirms."""
        from fitz_ai.governance import AnswerVerificationConstraint

        mock_fast = MagicMock()
        mock_balanced = MagicMock()
        # 2/3 fast say NO, balanced confirms NO
        mock_fast.chat.side_effect = ["NO", "NO", "YES"]
        mock_balanced.chat.return_value = "NO"
        constraint = AnswerVerificationConstraint(chat=mock_fast, chat_balanced=mock_balanced)

        chunks = [make_chunk("1", "France has 67 million people.")]
        result = constraint.apply("What is the capital of France?", chunks)

        assert result.allow_decisive_answer is False
        assert result.signal == "qualified"
        assert mock_balanced.chat.call_count == 1

    def test_jury_2no_balanced_rejects_allows(self):
        """Should allow when 2/3 fast say NO but balanced says YES (overrides)."""
        from fitz_ai.governance import AnswerVerificationConstraint

        mock_fast = MagicMock()
        mock_balanced = MagicMock()
        # 2/3 fast say NO, but balanced says YES (context is relevant)
        mock_fast.chat.side_effect = ["NO", "NO", "YES"]
        mock_balanced.chat.return_value = "YES"
        constraint = AnswerVerificationConstraint(chat=mock_fast, chat_balanced=mock_balanced)

        chunks = [make_chunk("1", "France is in Europe.")]
        result = constraint.apply("What is the capital of France?", chunks)

        assert result.allow_decisive_answer is True

    def test_jury_1_no_vote_allows(self):
        """Should allow when only 1/3 fast say NO (no balanced call needed)."""
        from fitz_ai.governance import AnswerVerificationConstraint

        mock_fast = MagicMock()
        mock_balanced = MagicMock()
        mock_fast.chat.side_effect = ["YES", "NO", "YES"]
        constraint = AnswerVerificationConstraint(chat=mock_fast, chat_balanced=mock_balanced)

        chunks = [make_chunk("1", "Paris is the capital of France.")]
        result = constraint.apply("What is the capital of France?", chunks)

        assert result.allow_decisive_answer is True
        # Balanced should NOT be called when <2 fast NO votes
        mock_balanced.chat.assert_not_called()

    def test_jury_error_counted_as_abstain(self):
        """Should handle LLM errors gracefully."""
        from fitz_ai.governance import AnswerVerificationConstraint

        mock_chat = MagicMock()
        # First two succeed, third errors - only 1 NO vote
        mock_chat.chat.side_effect = ["YES", "NO", Exception("LLM error")]
        constraint = AnswerVerificationConstraint(chat=mock_chat)

        chunks = [make_chunk("1", "Some content.")]
        result = constraint.apply("Any question?", chunks)

        assert result.allow_decisive_answer is True
