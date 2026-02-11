# tests/integration/test_governance_pipeline.py
"""Integration test for governance in the RAG pipeline."""


from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.governance import AnswerGovernor, ConstraintResult, run_constraints


class MockInsufficientEvidence:
    """Mock constraint that signals abstain when no chunks."""

    @property
    def name(self) -> str:
        return "insufficient_evidence"

    def apply(self, query, chunks):
        if not chunks:
            return ConstraintResult.deny(
                reason="No evidence retrieved",
                signal="abstain",
            )
        return ConstraintResult.allow()


class MockConflictAware:
    """Mock constraint that signals disputed when multiple chunks."""

    @property
    def name(self) -> str:
        return "conflict_aware"

    def apply(self, query, chunks):
        # Simulate conflict detection with multiple chunks
        if len(chunks) > 1:
            return ConstraintResult.deny(
                reason="Sources A and B conflict",
                signal="disputed",
            )
        return ConstraintResult.allow()


class TestGovernanceIntegration:
    """End-to-end governance flow tests."""

    def test_empty_chunks_triggers_abstain(self):
        """No chunks → insufficient evidence → ABSTAIN."""
        constraints = [MockInsufficientEvidence(), MockConflictAware()]
        governor = AnswerGovernor()

        results = run_constraints("What is X?", [], constraints)
        decision = governor.decide(results)

        assert decision.mode == AnswerMode.ABSTAIN
        assert "insufficient_evidence" in decision.triggered_constraints
        assert "No evidence" in decision.user_explanation

    def test_conflicting_chunks_triggers_disputed(self):
        """Multiple chunks with conflict → DISPUTED."""
        # Create mock chunks (simple dicts work for these mock constraints)
        mock_chunks = [{"id": "1"}, {"id": "2"}]

        constraints = [MockInsufficientEvidence(), MockConflictAware()]
        governor = AnswerGovernor()

        # MockInsufficientEvidence passes (chunks exist)
        # MockConflictAware triggers (multiple chunks)
        results = run_constraints("What is X?", mock_chunks, constraints)
        decision = governor.decide(results)

        assert decision.mode == AnswerMode.DISPUTED
        assert "conflict_aware" in decision.triggered_constraints

    def test_single_good_chunk_is_trustworthy(self):
        """Single chunk, no conflicts → TRUSTWORTHY."""
        mock_chunks = [{"id": "1"}]

        constraints = [MockInsufficientEvidence(), MockConflictAware()]
        governor = AnswerGovernor()

        results = run_constraints("What is X?", mock_chunks, constraints)
        decision = governor.decide(results)

        assert decision.mode == AnswerMode.TRUSTWORTHY
        assert decision.triggered_constraints == ()
        assert decision.user_explanation is None

    def test_abstain_wins_over_disputed(self):
        """When both signals present, ABSTAIN takes priority."""

        class AlwaysAbstain:
            name = "always_abstain"

            def apply(self, query, chunks):
                return ConstraintResult.deny(
                    reason="No evidence",
                    signal="abstain",
                )

        class AlwaysDisputed:
            name = "always_disputed"

            def apply(self, query, chunks):
                return ConstraintResult.deny(
                    reason="Sources conflict",
                    signal="disputed",
                )

        constraints = [AlwaysAbstain(), AlwaysDisputed()]
        governor = AnswerGovernor()

        results = run_constraints("query", [], constraints)
        decision = governor.decide(results)

        assert decision.mode == AnswerMode.ABSTAIN
        assert "abstain" in decision.signals
        assert "disputed" in decision.signals
        # Both constraints should be triggered
        assert len(decision.triggered_constraints) == 2


class TestGovernanceDecisionProperties:
    """Test GovernanceDecision helper properties."""

    def test_is_trustworthy_true_for_trustworthy_mode(self):
        """is_trustworthy returns True for TRUSTWORTHY mode."""
        governor = AnswerGovernor()
        decision = governor.decide([])

        assert decision.is_trustworthy is True

    def test_is_trustworthy_false_for_other_modes(self):
        """is_trustworthy returns False for non-TRUSTWORTHY modes."""
        governor = AnswerGovernor()

        result = ConstraintResult.deny(
            reason="test",
            signal="abstain",
        )
        # Add constraint_name to metadata
        result = ConstraintResult(
            allow_decisive_answer=False,
            reason="test",
            signal="abstain",
            metadata={"constraint_name": "test_constraint"},
        )

        decision = governor.decide([result])

        assert decision.is_trustworthy is False

    def test_should_include_caveats_for_disputed(self):
        """should_include_caveats is True for DISPUTED."""
        governor = AnswerGovernor()

        # DISPUTED
        result = ConstraintResult(
            allow_decisive_answer=False,
            reason="test",
            signal="disputed",
            metadata={"constraint_name": "test"},
        )
        decision = governor.decide([result])
        assert decision.should_include_caveats is True

    def test_should_not_include_caveats_for_trustworthy(self):
        """should_include_caveats is False for TRUSTWORTHY."""
        governor = AnswerGovernor()

        # Denial without signal → TRUSTWORTHY (no caveats)
        result = ConstraintResult(
            allow_decisive_answer=False,
            reason="test",
            signal=None,
            metadata={"constraint_name": "test"},
        )
        decision = governor.decide([result])
        assert decision.should_include_caveats is False

    def test_should_include_caveats_false_for_abstain(self):
        """ABSTAIN doesn't caveat, it refuses entirely."""
        governor = AnswerGovernor()

        result = ConstraintResult(
            allow_decisive_answer=False,
            reason="test",
            signal="abstain",
            metadata={"constraint_name": "test"},
        )
        decision = governor.decide([result])

        # ABSTAIN should NOT include caveats - it refuses to answer
        assert decision.should_include_caveats is False

    def test_to_dict_serialization(self):
        """GovernanceDecision serializes correctly to dict."""
        governor = AnswerGovernor()

        results = [
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Reason A",
                signal="abstain",
                metadata={"constraint_name": "constraint_a"},
            ),
            ConstraintResult(
                allow_decisive_answer=False,
                reason="Reason B",
                signal="disputed",
                metadata={"constraint_name": "constraint_b"},
            ),
        ]

        decision = governor.decide(results)
        d = decision.to_dict()

        assert d["mode"] == "abstain"
        assert "constraint_a" in d["triggered_constraints"]
        assert "constraint_b" in d["triggered_constraints"]
        assert "Reason A" in d["reasons"]
        assert "Reason B" in d["reasons"]
        assert "abstain" in d["signals"]
        assert "disputed" in d["signals"]
