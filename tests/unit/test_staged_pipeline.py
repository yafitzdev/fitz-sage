# tests/unit/test_staged_pipeline.py
"""Tests for staged constraint pipeline."""


from fitz_ai.governance.constraints.base import ConstraintResult
from fitz_ai.governance.constraints.staged import (
    ConstraintStage,
    StageContext,
    StagedConstraintPipeline,
    _build_staged_pipeline,
    run_staged_constraints,
)


class MockConstraint:
    """Test constraint with configurable behavior."""

    def __init__(
        self,
        name: str,
        allow: bool = True,
        signal: str | None = None,
        metadata: dict | None = None,
    ):
        self._name = name
        self._allow = allow
        self._signal = signal
        self._metadata = metadata or {}
        self.called = False

    @property
    def name(self) -> str:
        return self._name

    def apply(self, query: str, chunks) -> ConstraintResult:
        self.called = True
        if self._allow:
            return ConstraintResult.allow()
        return ConstraintResult.deny(
            reason=f"{self._name} denied",
            signal=self._signal,
            **self._metadata,
        )


class CrashingConstraint:
    """Constraint that always raises."""

    @property
    def name(self) -> str:
        return "crasher"

    def apply(self, query, chunks):
        raise RuntimeError("Boom!")


class TestStagedConstraintPipeline:
    """Tests for the StagedConstraintPipeline class."""

    def test_stages_execute_in_order(self):
        """Stages run in declaration order."""
        call_order = []

        class OrderTracker:
            def __init__(self, n, stage_label):
                self._name = n
                self._label = stage_label

            @property
            def name(self):
                return self._name

            def apply(self, query, chunks):
                call_order.append(self._label)
                return ConstraintResult.allow()

        pipeline = StagedConstraintPipeline(
            stages=[
                ConstraintStage(
                    name="relevance",
                    constraints=[OrderTracker("insufficient_evidence", "s1")],
                ),
                ConstraintStage(
                    name="sufficiency",
                    constraints=[OrderTracker("causal_attribution", "s2")],
                ),
                ConstraintStage(
                    name="consistency",
                    constraints=[OrderTracker("conflict_aware", "s3")],
                ),
            ]
        )

        pipeline.run("test query", [])
        assert call_order == ["s1", "s2", "s3"]

    def test_abstain_short_circuits_consistency(self):
        """When Stage 1 signals abstain, Stage 3 is skipped entirely."""
        ie = MockConstraint("insufficient_evidence", allow=False, signal="abstain")
        ca = MockConstraint("conflict_aware", allow=False, signal="disputed")

        pipeline = StagedConstraintPipeline(
            stages=[
                ConstraintStage(
                    name="relevance",
                    constraints=[ie],
                    short_circuit_signals=frozenset({"abstain"}),
                ),
                ConstraintStage(
                    name="consistency",
                    constraints=[ca],
                ),
            ]
        )

        results = pipeline.run("test", [])

        assert ie.called is True
        assert ca.called is False
        assert len(results) == 1
        assert results[0].signal == "abstain"

    def test_abstain_short_circuits_sufficiency_and_consistency(self):
        """Abstain skips both sufficiency and consistency stages."""
        ie = MockConstraint("insufficient_evidence", allow=False, signal="abstain")
        causal = MockConstraint("causal_attribution", allow=False, signal="qualified")
        ca = MockConstraint("conflict_aware", allow=False, signal="disputed")

        pipeline = StagedConstraintPipeline(
            stages=[
                ConstraintStage(
                    name="relevance",
                    constraints=[ie],
                    short_circuit_signals=frozenset({"abstain"}),
                ),
                ConstraintStage(
                    name="sufficiency",
                    constraints=[causal],
                ),
                ConstraintStage(
                    name="consistency",
                    constraints=[ca],
                ),
            ]
        )

        results = pipeline.run("test", [])

        assert ie.called is True
        assert causal.called is False
        assert ca.called is False
        assert len(results) == 1

    def test_qualified_does_not_short_circuit(self):
        """When Stage 1 signals qualified, all stages still run."""
        ie = MockConstraint(
            "insufficient_evidence",
            allow=False,
            signal="qualified",
            metadata={"max_similarity": 0.65},
        )
        ca = MockConstraint("conflict_aware", allow=True)

        pipeline = StagedConstraintPipeline(
            stages=[
                ConstraintStage(
                    name="relevance",
                    constraints=[ie],
                    short_circuit_signals=frozenset({"abstain"}),
                ),
                ConstraintStage(
                    name="consistency",
                    constraints=[ca],
                ),
            ]
        )

        results = pipeline.run("test", [])

        assert ie.called is True
        assert ca.called is True
        assert len(results) == 2

    def test_all_allow_returns_all_results(self):
        """When everything passes, all results returned."""
        c1 = MockConstraint("insufficient_evidence", allow=True)
        c2 = MockConstraint("causal_attribution", allow=True)
        c3 = MockConstraint("conflict_aware", allow=True)

        pipeline = StagedConstraintPipeline(
            stages=[
                ConstraintStage(name="relevance", constraints=[c1]),
                ConstraintStage(name="sufficiency", constraints=[c2]),
                ConstraintStage(name="consistency", constraints=[c3]),
            ]
        )

        results = pipeline.run("test", [])

        assert len(results) == 3
        assert all(r.allow_decisive_answer for r in results)

    def test_result_metadata_includes_stage(self):
        """Denied results carry stage name in metadata."""
        c = MockConstraint("conflict_aware", allow=False, signal="disputed")

        pipeline = StagedConstraintPipeline(
            stages=[ConstraintStage(name="consistency", constraints=[c])]
        )

        results = pipeline.run("test", [])

        assert len(results) == 1
        assert results[0].metadata["stage"] == "consistency"
        assert results[0].metadata["constraint_name"] == "conflict_aware"

    def test_allow_results_have_stage_metadata(self):
        """All results (allow and deny) get stage metadata for feature extraction."""
        c = MockConstraint("insufficient_evidence", allow=True)

        pipeline = StagedConstraintPipeline(
            stages=[ConstraintStage(name="relevance", constraints=[c])]
        )

        results = pipeline.run("test", [])

        assert len(results) == 1
        assert results[0].metadata["stage"] == "relevance"
        assert results[0].metadata["constraint_name"] == "insufficient_evidence"

    def test_crashing_constraint_skipped(self):
        """Exception in a constraint is caught, other constraints still run."""
        crasher = CrashingConstraint()
        safe = MockConstraint("safe_constraint", allow=True)

        pipeline = StagedConstraintPipeline(
            stages=[ConstraintStage(name="sufficiency", constraints=[crasher, safe])]
        )

        results = pipeline.run("test", [])

        assert len(results) == 1
        assert results[0].allow_decisive_answer is True

    def test_empty_stages_returns_empty(self):
        """Pipeline with no stages returns empty results."""
        pipeline = StagedConstraintPipeline(stages=[])
        results = pipeline.run("test", [])
        assert results == []


class TestStageContext:
    """Tests for StageContext propagation."""

    def test_relevance_stage_updates_context(self):
        """Stage 1 results update context for downstream stages."""
        ie = MockConstraint(
            "insufficient_evidence",
            allow=False,
            signal="qualified",
            metadata={"ie_max_similarity": 0.55},
        )

        pipeline = StagedConstraintPipeline(
            stages=[
                ConstraintStage(
                    name="relevance",
                    constraints=[ie],
                    short_circuit_signals=frozenset({"abstain"}),
                ),
            ]
        )

        # We need to inspect context — run manually
        context = StageContext()
        stage_results = pipeline._run_stage("test", [], pipeline.stages[0])
        pipeline._update_context(pipeline.stages[0], stage_results, context)

        assert context.relevance_confirmed is False
        assert context.relevance_signal == "qualified"
        assert context.max_similarity == 0.55

    def test_sufficiency_stage_updates_context(self):
        """Stage 2 results track evidence gaps."""
        causal = MockConstraint("causal_attribution", allow=False, signal="qualified")

        pipeline = StagedConstraintPipeline(
            stages=[ConstraintStage(name="sufficiency", constraints=[causal])]
        )

        context = StageContext()
        stage_results = pipeline._run_stage("test", [], pipeline.stages[0])
        pipeline._update_context(pipeline.stages[0], stage_results, context)

        assert context.sufficiency_checked is True
        assert context.has_sufficient_evidence is False
        assert len(context.evidence_gaps) == 1

    def test_all_allow_leaves_context_default(self):
        """When all constraints allow, context stays at defaults."""
        c = MockConstraint("insufficient_evidence", allow=True)

        pipeline = StagedConstraintPipeline(
            stages=[ConstraintStage(name="relevance", constraints=[c])]
        )

        context = StageContext()
        stage_results = pipeline._run_stage("test", [], pipeline.stages[0])
        pipeline._update_context(pipeline.stages[0], stage_results, context)

        assert context.relevance_confirmed is True
        assert context.relevance_signal is None


class TestBuildStagedPipeline:
    """Tests for auto-classification of constraints into stages."""

    def test_known_constraints_classified_correctly(self):
        """Constraints are grouped by name into correct stages."""
        constraints = [
            MockConstraint("insufficient_evidence"),
            MockConstraint("specific_info_type"),
            MockConstraint("causal_attribution"),
            MockConstraint("answer_verification"),
            MockConstraint("conflict_aware"),
        ]

        pipeline = _build_staged_pipeline(constraints)

        assert len(pipeline.stages) == 3
        assert pipeline.stages[0].name == "relevance"
        assert len(pipeline.stages[0].constraints) == 1
        assert pipeline.stages[1].name == "sufficiency"
        assert len(pipeline.stages[1].constraints) == 3
        assert pipeline.stages[2].name == "consistency"
        assert len(pipeline.stages[2].constraints) == 1

    def test_unknown_constraint_defaults_to_sufficiency(self):
        """Unrecognized constraint names go to sufficiency stage."""
        constraints = [
            MockConstraint("my_custom_constraint"),
        ]

        pipeline = _build_staged_pipeline(constraints)

        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].name == "sufficiency"

    def test_relevance_stage_has_short_circuit(self):
        """Relevance stage is configured with abstain short-circuit."""
        constraints = [
            MockConstraint("insufficient_evidence"),
            MockConstraint("conflict_aware"),
        ]

        pipeline = _build_staged_pipeline(constraints)

        relevance_stage = pipeline.stages[0]
        assert relevance_stage.name == "relevance"
        assert "abstain" in relevance_stage.short_circuit_signals

    def test_empty_constraints_returns_empty_pipeline(self):
        """No constraints produces pipeline with no stages."""
        pipeline = _build_staged_pipeline([])
        assert len(pipeline.stages) == 0


class TestRunStagedConstraints:
    """Tests for the drop-in replacement function."""

    def test_empty_returns_empty(self):
        """No constraints means empty results."""
        results = run_staged_constraints("query", [], [])
        assert results == []

    def test_all_allow(self):
        """All-allow constraints produce allow results."""
        constraints = [
            MockConstraint("insufficient_evidence", allow=True),
            MockConstraint("conflict_aware", allow=True),
        ]

        results = run_staged_constraints("query", [], constraints)

        assert len(results) == 2
        assert all(r.allow_decisive_answer for r in results)

    def test_abstain_prevents_conflict_detection(self):
        """Abstain from IE prevents ConflictAware from running."""
        ie = MockConstraint("insufficient_evidence", allow=False, signal="abstain")
        ca = MockConstraint("conflict_aware", allow=False, signal="disputed")

        results = run_staged_constraints("query", [], [ie, ca])

        assert ie.called is True
        assert ca.called is False
        assert len(results) == 1
        assert results[0].signal == "abstain"

    def test_backward_compatible_with_unknown_constraints(self):
        """Unknown constraints all run (no short-circuiting between them)."""
        c1 = MockConstraint("c1", allow=True)
        c2 = MockConstraint("c2", allow=False, signal="disputed")
        c3 = MockConstraint("c3", allow=True)

        results = run_staged_constraints("query", [], [c1, c2, c3])

        assert c1.called is True
        assert c2.called is True
        assert c3.called is True
        assert len(results) == 3
