# tests/unit/test_answer_mode.py
"""
Tests for the AnswerMode system.

These tests verify:
1. AnswerMode enum values
2. AnswerGovernor decision logic
3. Mode instruction mapping
"""

from __future__ import annotations

from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.governance import AnswerGovernor, ConstraintResult
from fitz_ai.governance.instructions import (
    MODE_INSTRUCTIONS,
    get_mode_instruction,
)

# =============================================================================
# Tests: AnswerMode Enum
# =============================================================================


class TestAnswerModeEnum:
    """Test AnswerMode enum."""

    def test_all_modes_defined(self):
        """Should have all expected modes."""
        assert AnswerMode.TRUSTWORTHY == "trustworthy"
        assert AnswerMode.DISPUTED == "disputed"
        assert AnswerMode.ABSTAIN == "abstain"

    def test_mode_is_string(self):
        """Modes should be string-compatible."""
        assert isinstance(AnswerMode.TRUSTWORTHY, str)
        assert AnswerMode.TRUSTWORTHY.value == "trustworthy"


# =============================================================================
# Tests: AnswerGovernor
# =============================================================================


class TestAnswerGovernor:
    """Test the answer mode resolution logic via AnswerGovernor."""

    def test_empty_results_returns_trustworthy(self):
        """Should return TRUSTWORTHY when no constraint results."""
        governor = AnswerGovernor()
        decision = governor.decide([])
        assert decision.mode == AnswerMode.TRUSTWORTHY

    def test_all_allowed_returns_trustworthy(self):
        """Should return TRUSTWORTHY when all constraints allow."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult.allow(),
            ConstraintResult.allow(),
        ]
        decision = governor.decide(results)
        assert decision.mode == AnswerMode.TRUSTWORTHY

    def test_abstain_signal_returns_abstain(self):
        """Should return ABSTAIN when abstain signal present."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult.allow(),
            ConstraintResult(
                allow_decisive_answer=False,
                reason="No evidence",
                signal="abstain",
                metadata={"constraint_name": "insufficient_evidence"},
            ),
        ]
        decision = governor.decide(results)
        assert decision.mode == AnswerMode.ABSTAIN

    def test_disputed_signal_returns_disputed(self):
        """Should return DISPUTED when disputed signal present."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult.allow(),
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Conflicting sources",
                signal="disputed",
                metadata={"constraint_name": "conflict_aware"},
            ),
        ]
        decision = governor.decide(results)
        assert decision.mode == AnswerMode.DISPUTED

    def test_abstain_takes_priority_over_disputed(self):
        """ABSTAIN should take priority over DISPUTED."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Conflict",
                signal="disputed",
                metadata={"constraint_name": "conflict_aware"},
            ),
            ConstraintResult(
                allow_decisive_answer=False,
                reason="No evidence",
                signal="abstain",
                metadata={"constraint_name": "insufficient_evidence"},
            ),
        ]
        decision = governor.decide(results)
        assert decision.mode == AnswerMode.ABSTAIN

    def test_denial_without_signal_returns_trustworthy(self):
        """Should return TRUSTWORTHY when denied without recognized signal."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Some issue",
                signal=None,
                metadata={"constraint_name": "custom"},
            ),
        ]
        decision = governor.decide(results)
        assert decision.mode == AnswerMode.TRUSTWORTHY


# =============================================================================
# Tests: Mode Instructions
# =============================================================================


class TestModeInstructions:
    """Test mode instruction mapping."""

    def test_all_modes_have_instructions(self):
        """Every mode should have an instruction."""
        for mode in AnswerMode:
            assert mode in MODE_INSTRUCTIONS
            assert len(MODE_INSTRUCTIONS[mode]) > 0

    def test_get_mode_instruction(self):
        """get_mode_instruction should return correct instruction."""
        instruction = get_mode_instruction(AnswerMode.DISPUTED)
        assert "disagree" in instruction.lower()

        instruction = get_mode_instruction(AnswerMode.ABSTAIN)
        assert "definitive" in instruction.lower()

    def test_trustworthy_instruction_is_direct(self):
        """TRUSTWORTHY instruction should be direct."""
        instruction = get_mode_instruction(AnswerMode.TRUSTWORTHY)
        assert "clearly" in instruction.lower() or "directly" in instruction.lower()


# =============================================================================
# Tests: Integration (Constraint → Mode)
# =============================================================================


class TestConstraintToModeIntegration:
    """Test the full flow from constraint results to mode."""

    def test_conflict_constraint_produces_disputed(self):
        """ConflictAwareConstraint denial should produce DISPUTED mode."""
        governor = AnswerGovernor()
        # Simulate what ConflictAwareConstraint returns
        result = ConstraintResult(
            allow_decisive_answer=False,
            reason="Conflicting classifications detected",
            signal="disputed",
            metadata={"constraint_name": "conflict_aware"},
        )

        decision = governor.decide([result])
        assert decision.mode == AnswerMode.DISPUTED

    def test_insufficient_evidence_produces_abstain(self):
        """InsufficientEvidenceConstraint denial should produce ABSTAIN mode."""
        governor = AnswerGovernor()
        # Simulate what InsufficientEvidenceConstraint returns
        result = ConstraintResult(
            allow_decisive_answer=False,
            reason="No explicit causal evidence found",
            signal="abstain",
            metadata={"constraint_name": "insufficient_evidence"},
        )

        decision = governor.decide([result])
        assert decision.mode == AnswerMode.ABSTAIN

    def test_both_constraints_deny_abstain_wins(self):
        """When both constraints deny, ABSTAIN should take priority."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Conflict",
                signal="disputed",
                metadata={"constraint_name": "conflict_aware"},
            ),
            ConstraintResult(
                allow_decisive_answer=False,
                reason="No evidence",
                signal="abstain",
                metadata={"constraint_name": "insufficient_evidence"},
            ),
        ]

        decision = governor.decide(results)
        assert decision.mode == AnswerMode.ABSTAIN
