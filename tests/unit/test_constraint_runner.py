# tests/unit/test_constraint_runner.py
"""Tests for constraint runner."""


from fitz_ai.governance.constraints.base import ConstraintResult
from fitz_ai.governance.constraints.runner import run_constraints


class MockConstraint:
    """Test constraint that can be configured to allow or deny."""

    def __init__(self, name: str, allow: bool = True, signal: str | None = None):
        self._name = name
        self._allow = allow
        self._signal = signal

    @property
    def name(self) -> str:
        return self._name

    def apply(self, query: str, chunks) -> ConstraintResult:
        if self._allow:
            return ConstraintResult.allow()
        return ConstraintResult.deny(
            reason=f"{self._name} denied",
            signal=self._signal,
        )


class TestRunConstraints:
    """Tests for run_constraints function."""

    def test_empty_constraints_returns_empty_list(self):
        """No constraints means empty results."""
        results = run_constraints("query", [], [])
        assert results == []

    def test_all_pass_returns_all_results(self):
        """Each constraint produces a result."""
        constraints = [
            MockConstraint("c1", allow=True),
            MockConstraint("c2", allow=True),
        ]

        results = run_constraints("query", [], constraints)

        assert len(results) == 2
        assert all(r.allow_decisive_answer for r in results)

    def test_denials_preserve_signals(self):
        """Individual signals are preserved (not combined)."""
        constraints = [
            MockConstraint("c1", allow=False, signal="abstain"),
            MockConstraint("c2", allow=False, signal="disputed"),
        ]

        results = run_constraints("query", [], constraints)

        assert len(results) == 2
        signals = {r.signal for r in results}
        assert signals == {"abstain", "disputed"}

    def test_constraint_names_injected(self):
        """Constraint names are added to metadata for denied results."""
        constraints = [
            MockConstraint("my_constraint", allow=False, signal="abstain"),
        ]

        results = run_constraints("query", [], constraints)

        assert len(results) == 1
        assert results[0].metadata["constraint_name"] == "my_constraint"

    def test_allowing_constraints_have_metadata(self):
        """All results (allow and deny) get constraint_name and stage metadata."""
        constraints = [
            MockConstraint("my_constraint", allow=True),
        ]

        results = run_constraints("query", [], constraints)

        assert len(results) == 1
        assert results[0].allow_decisive_answer is True
        assert results[0].metadata["constraint_name"] == "my_constraint"
        assert "stage" in results[0].metadata

    def test_exception_skips_constraint(self):
        """Crashing constraints are skipped (fail-safe)."""

        class CrashingConstraint:
            @property
            def name(self) -> str:
                return "crasher"

            def apply(self, query, chunks):
                raise RuntimeError("Boom!")

        constraints = [
            CrashingConstraint(),
            MockConstraint("safe", allow=True),
        ]

        results = run_constraints("query", [], constraints)

        # Only the safe constraint's result
        assert len(results) == 1
        assert results[0].allow_decisive_answer is True

    def test_mixed_allow_and_deny(self):
        """Mixed results preserve all outcomes."""
        constraints = [
            MockConstraint("c1", allow=True),
            MockConstraint("c2", allow=False, signal="disputed"),
            MockConstraint("c3", allow=True),
            MockConstraint("c4", allow=False, signal="abstain"),
        ]

        results = run_constraints("query", [], constraints)

        assert len(results) == 4

        allow_count = sum(1 for r in results if r.allow_decisive_answer)
        deny_count = sum(1 for r in results if not r.allow_decisive_answer)

        assert allow_count == 2
        assert deny_count == 2

    def test_reason_preserved(self):
        """Denial reasons are preserved in results."""
        constraints = [
            MockConstraint("custom", allow=False, signal="abstain"),
        ]

        results = run_constraints("query", [], constraints)

        assert len(results) == 1
        assert "custom denied" in results[0].reason
