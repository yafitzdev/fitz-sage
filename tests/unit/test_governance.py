# tests/unit/test_governance.py
"""Tests for answer governance."""

import pytest

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.governance import AnswerGovernor, GovernanceDecision
from fitz_sage.governance.constraints.base import ConstraintResult


class TestGovernanceDecision:
    """Tests for GovernanceDecision dataclass."""

    def test_trustworthy_decision(self):
        """Trustworthy decisions have no explanations."""
        decision = GovernanceDecision.trustworthy()

        assert decision.mode == AnswerMode.TRUSTWORTHY
        assert decision.is_trustworthy is True
        assert decision.should_include_caveats is False
        assert decision.user_explanation is None
        assert decision.triggered_constraints == ()

    def test_abstain_decision_with_reasons(self):
        """Abstain decisions include reasons."""
        decision = GovernanceDecision(
            mode=AnswerMode.ABSTAIN,
            triggered_constraints=("insufficient_evidence",),
            reasons=("No evidence retrieved",),
            signals=frozenset({"abstain"}),
        )

        assert decision.mode == AnswerMode.ABSTAIN
        assert decision.is_trustworthy is False
        # ABSTAIN doesn't caveat, it refuses
        assert decision.should_include_caveats is False
        assert decision.user_explanation == "No evidence retrieved"

    def test_disputed_decision(self):
        """Disputed decisions surface conflict."""
        decision = GovernanceDecision(
            mode=AnswerMode.DISPUTED,
            triggered_constraints=("conflict_aware",),
            reasons=("Sources A and B contradict",),
            signals=frozenset({"disputed"}),
        )

        assert decision.mode == AnswerMode.DISPUTED
        assert decision.should_include_caveats is True
        assert "conflicting" in decision.user_explanation.lower()

    def test_to_dict_serialization(self):
        """Decisions serialize to dict for logging."""
        decision = GovernanceDecision(
            mode=AnswerMode.ABSTAIN,
            triggered_constraints=("a", "b"),
            reasons=("reason1", "reason2"),
            signals=frozenset({"abstain"}),
        )

        d = decision.to_dict()

        assert d["mode"] == "abstain"
        assert d["triggered_constraints"] == ["a", "b"]
        assert d["reasons"] == ["reason1", "reason2"]
        assert "abstain" in d["signals"]

    def test_abstain_without_reasons_has_default_explanation(self):
        """Abstain mode provides default explanation when no reasons."""
        decision = GovernanceDecision(
            mode=AnswerMode.ABSTAIN,
            signals=frozenset({"abstain"}),
        )

        assert "don't have enough information" in decision.user_explanation

    def test_disputed_without_reasons_has_default_explanation(self):
        """Disputed mode provides default explanation when no reasons."""
        decision = GovernanceDecision(
            mode=AnswerMode.DISPUTED,
            signals=frozenset({"disputed"}),
        )

        assert "conflicting information" in decision.user_explanation.lower()


class TestAnswerGovernor:
    """Tests for AnswerGovernor decision logic."""

    def test_empty_results_returns_trustworthy(self):
        """No constraints means trustworthy."""
        governor = AnswerGovernor()
        decision = governor.decide([])

        assert decision.mode == AnswerMode.TRUSTWORTHY

    def test_all_allow_returns_trustworthy(self):
        """All constraints passing means trustworthy."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult.allow(),
            ConstraintResult.allow(),
        ]

        decision = governor.decide(results)

        assert decision.mode == AnswerMode.TRUSTWORTHY
        assert decision.triggered_constraints == ()

    def test_abstain_signal_takes_priority(self):
        """Abstain signal wins over disputed."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult(
                allow_decisive_answer=False,
                reason="No evidence",
                signal="abstain",
                metadata={"constraint_name": "insufficient_evidence"},
            ),
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Sources conflict",
                signal="disputed",
                metadata={"constraint_name": "conflict_aware"},
            ),
        ]

        decision = governor.decide(results)

        assert decision.mode == AnswerMode.ABSTAIN
        assert "abstain" in decision.signals
        assert "disputed" in decision.signals
        assert len(decision.triggered_constraints) == 2

    def test_disputed_wins_over_plain_denial(self):
        """Disputed signal wins over plain denial."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Sources conflict",
                signal="disputed",
                metadata={"constraint_name": "conflict_aware"},
            ),
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Some other issue",
                signal=None,
                metadata={"constraint_name": "custom_constraint"},
            ),
        ]

        decision = governor.decide(results)

        assert decision.mode == AnswerMode.DISPUTED

    def test_denial_without_signal_becomes_trustworthy(self):
        """Denial without recognized signal resolves to TRUSTWORTHY."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Some limitation",
                signal=None,
                metadata={"constraint_name": "my_constraint"},
            ),
        ]

        decision = governor.decide(results)

        assert decision.mode == AnswerMode.TRUSTWORTHY
        assert "my_constraint" in decision.triggered_constraints

    def test_constraint_names_tracked(self):
        """Triggered constraint names are preserved."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult.allow(),
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Issue A",
                signal="abstain",
                metadata={"constraint_name": "constraint_a"},
            ),
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Issue B",
                signal="qualified",
                metadata={"constraint_name": "constraint_b"},
            ),
        ]

        decision = governor.decide(results)

        assert "constraint_a" in decision.triggered_constraints
        assert "constraint_b" in decision.triggered_constraints
        # The allowing constraint should not appear
        assert len(decision.triggered_constraints) == 2

    def test_reasons_collected(self):
        """All denial reasons are collected."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Reason one",
                signal="abstain",
                metadata={"constraint_name": "c1"},
            ),
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Reason two",
                signal="abstain",
                metadata={"constraint_name": "c2"},
            ),
        ]

        decision = governor.decide(results)

        assert "Reason one" in decision.reasons
        assert "Reason two" in decision.reasons

    def test_missing_constraint_name_uses_unknown(self):
        """Missing constraint_name in metadata defaults to 'unknown'."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Some issue",
                signal="abstain",
                metadata={},  # No constraint_name
            ),
        ]

        decision = governor.decide(results)

        assert "unknown" in decision.triggered_constraints


class TestSignalPriority:
    """Tests specifically for signal priority ordering."""

    @pytest.mark.parametrize(
        "signals,expected_mode",
        [
            (["abstain"], AnswerMode.ABSTAIN),
            (["disputed"], AnswerMode.DISPUTED),
            (["qualified"], AnswerMode.TRUSTWORTHY),
            (["abstain", "disputed"], AnswerMode.ABSTAIN),
            (["abstain", "qualified"], AnswerMode.ABSTAIN),
            (["disputed", "qualified"], AnswerMode.DISPUTED),
            (["abstain", "disputed", "qualified"], AnswerMode.ABSTAIN),
        ],
    )
    def test_signal_priority(self, signals, expected_mode):
        """Signal priority: abstain > disputed > everything else → trustworthy."""
        governor = AnswerGovernor()
        results = [
            ConstraintResult(
                allow_decisive_answer=False,
                reason=f"Reason for {sig}",
                signal=sig,
                metadata={"constraint_name": f"constraint_{sig}"},
            )
            for sig in signals
        ]

        decision = governor.decide(results)

        assert decision.mode == expected_mode
