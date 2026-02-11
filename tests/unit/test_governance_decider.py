# tests/unit/test_governance_decider.py
"""Tests for GovernanceDecider ML-based governance decision maker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.tier1

from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.engines.fitz_rag.governance import GovernanceDecision
from fitz_ai.engines.fitz_rag.guardrails.base import ConstraintResult
from fitz_ai.engines.fitz_rag.guardrails.governance_decider import GovernanceDecider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_constraint_result(
    allow: bool = True,
    signal: str | None = None,
    reason: str | None = None,
    name: str = "test_constraint",
) -> ConstraintResult:
    return ConstraintResult(
        allow_decisive_answer=allow,
        signal=signal,
        reason=reason,
        metadata={"constraint_name": name},
    )


def _make_mock_artifact():
    """Create a mock model artifact dict matching the real joblib structure."""
    import numpy as np

    # Stage 1 mock: always predicts answerable
    s1_model = MagicMock()
    s1_model.classes_ = np.array(["abstain", "answerable"])
    s1_model.predict_proba.return_value = np.array([[0.1, 0.9]])

    # Stage 2 mock: predicts trustworthy by default
    s2_model = MagicMock()
    s2_model.classes_ = np.array(["disputed", "trustworthy"])
    s2_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    return {
        "stage1_model": s1_model,
        "stage2_model": s2_model,
        "stage1_threshold": 0.5,
        "stage2_threshold": 0.75,
        "encoders": {},
        "feature_names": ["num_constraints_fired", "query_word_count", "num_chunks"],
    }


# ---------------------------------------------------------------------------
# Test: Fallback behavior (no model)
# ---------------------------------------------------------------------------


class TestFallbackBehavior:
    def test_unavailable_without_model(self):
        decider = GovernanceDecider(model_path=Path("nonexistent_model.joblib"))
        assert not decider.available

    def test_falls_back_to_governor_when_no_model(self):
        decider = GovernanceDecider(model_path=Path("nonexistent_model.joblib"))
        results = [_make_constraint_result(allow=True)]
        decision = decider.decide(results, {"num_chunks": 3})
        assert isinstance(decision, GovernanceDecision)
        assert decision.mode == AnswerMode.TRUSTWORTHY

    def test_falls_back_when_features_none(self):
        decider = GovernanceDecider(model_path=Path("nonexistent_model.joblib"))
        results = [_make_constraint_result(allow=False, signal="disputed", name="conflict_aware")]
        decision = decider.decide(results, None)
        assert isinstance(decision, GovernanceDecision)

    def test_falls_back_when_features_empty(self):
        """Empty features dict should trigger fallback."""
        decider = GovernanceDecider(model_path=Path("nonexistent_model.joblib"))
        results = [_make_constraint_result(allow=True)]
        decision = decider.decide(results, {})
        assert isinstance(decision, GovernanceDecision)


# ---------------------------------------------------------------------------
# Test: ML prediction
# ---------------------------------------------------------------------------


class TestMLPrediction:
    @pytest.fixture
    def decider(self):
        artifact = _make_mock_artifact()
        with patch(
            "fitz_ai.engines.fitz_rag.guardrails.governance_decider.GovernanceDecider.__init__",
            lambda self, **kw: None,
        ):
            d = GovernanceDecider.__new__(GovernanceDecider)
        from fitz_ai.engines.fitz_rag.governance import AnswerGovernor

        d._available = True
        d._governor = AnswerGovernor()
        d._s1_model = artifact["stage1_model"]
        d._s2_model = artifact["stage2_model"]
        d._s1_threshold = artifact["stage1_threshold"]
        d._s2_threshold = artifact["stage2_threshold"]
        d._encoders = artifact["encoders"]
        d._feature_names = artifact["feature_names"]
        return d

    def test_trustworthy_no_constraints_returns_trustworthy(self, decider):
        import numpy as np

        decider._s2_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        results = [_make_constraint_result(allow=True)]
        features = {"num_constraints_fired": 0, "query_word_count": 5, "num_chunks": 3}
        decision = decider.decide(results, features)
        assert decision.mode == AnswerMode.TRUSTWORTHY

    def test_trustworthy_with_constraints_returns_trustworthy(self, decider):
        import numpy as np

        decider._s2_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        results = [
            _make_constraint_result(allow=False, signal="qualified", reason="hedged", name="ie")
        ]
        features = {"num_constraints_fired": 1, "query_word_count": 5, "num_chunks": 3}
        decision = decider.decide(results, features)
        assert decision.mode == AnswerMode.TRUSTWORTHY
        assert "ie" in decision.triggered_constraints
        assert "hedged" in decision.reasons

    def test_disputed_prediction(self, decider):
        import numpy as np

        decider._s2_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        results = [_make_constraint_result(allow=False, signal="disputed", name="conflict_aware")]
        features = {"num_constraints_fired": 1, "query_word_count": 5, "num_chunks": 3}
        decision = decider.decide(results, features)
        assert decision.mode == AnswerMode.DISPUTED

    def test_abstain_prediction(self, decider):
        import numpy as np

        decider._s1_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        results = [
            _make_constraint_result(allow=False, signal="abstain", name="insufficient_evidence")
        ]
        features = {"num_constraints_fired": 1, "query_word_count": 5, "num_chunks": 3}
        decision = decider.decide(results, features)
        assert decision.mode == AnswerMode.ABSTAIN


# ---------------------------------------------------------------------------
# Test: Feature preparation
# ---------------------------------------------------------------------------


pd = pytest.importorskip("pandas", reason="pandas required for feature preparation tests")


class TestFeaturePreparation:
    @pytest.fixture
    def decider(self):
        artifact = _make_mock_artifact()
        with patch(
            "fitz_ai.engines.fitz_rag.guardrails.governance_decider.GovernanceDecider.__init__",
            lambda self, **kw: None,
        ):
            d = GovernanceDecider.__new__(GovernanceDecider)
        from fitz_ai.engines.fitz_rag.governance import AnswerGovernor

        d._available = True
        d._governor = AnswerGovernor()
        d._s1_model = artifact["stage1_model"]
        d._s2_model = artifact["stage2_model"]
        d._s1_threshold = artifact["stage1_threshold"]
        d._s2_threshold = artifact["stage2_threshold"]
        d._encoders = artifact["encoders"]
        d._feature_names = artifact["feature_names"]
        return d

    def test_missing_features_filled_with_zero(self, decider):
        features = {"query_word_count": 5}
        X = decider._prepare_features(features)
        assert list(X.columns) == decider._feature_names
        assert X["num_constraints_fired"].iloc[0] == 0
        assert X["num_chunks"].iloc[0] == 0

    def test_extra_features_dropped(self, decider):
        features = {
            "num_constraints_fired": 1,
            "query_word_count": 5,
            "num_chunks": 3,
            "extra_feature": 999,
        }
        X = decider._prepare_features(features)
        assert "extra_feature" not in X.columns
        assert list(X.columns) == decider._feature_names

    def test_bool_features_converted(self, decider):
        decider._feature_names = ["ie_fired", "ca_fired"]
        features = {"ie_fired": True, "ca_fired": False}
        X = decider._prepare_features(features)
        assert X["ie_fired"].iloc[0] == 1
        assert X["ca_fired"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Test: AnswerMode mapping
# ---------------------------------------------------------------------------


class TestAnswerModeMapping:
    def test_abstain_maps_directly(self):
        assert GovernanceDecider._map_to_answer_mode("abstain", []) == AnswerMode.ABSTAIN

    def test_disputed_maps_directly(self):
        assert GovernanceDecider._map_to_answer_mode("disputed", []) == AnswerMode.DISPUTED

    def test_trustworthy_no_triggers_is_trustworthy(self):
        assert GovernanceDecider._map_to_answer_mode("trustworthy", []) == AnswerMode.TRUSTWORTHY

    def test_trustworthy_with_triggers_is_trustworthy(self):
        result = GovernanceDecider._map_to_answer_mode("trustworthy", ["conflict_aware"])
        assert result == AnswerMode.TRUSTWORTHY


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.fixture
    def decider(self):
        artifact = _make_mock_artifact()
        with patch(
            "fitz_ai.engines.fitz_rag.guardrails.governance_decider.GovernanceDecider.__init__",
            lambda self, **kw: None,
        ):
            d = GovernanceDecider.__new__(GovernanceDecider)
        from fitz_ai.engines.fitz_rag.governance import AnswerGovernor

        d._available = True
        d._governor = AnswerGovernor()
        d._s1_model = artifact["stage1_model"]
        d._s2_model = artifact["stage2_model"]
        d._s1_threshold = artifact["stage1_threshold"]
        d._s2_threshold = artifact["stage2_threshold"]
        d._encoders = artifact["encoders"]
        d._feature_names = artifact["feature_names"]
        return d

    def test_prediction_error_falls_back(self, decider):
        decider._s1_model.predict_proba.side_effect = RuntimeError("model error")
        results = [_make_constraint_result(allow=True)]
        features = {"num_constraints_fired": 0, "query_word_count": 5, "num_chunks": 3}
        decision = decider.decide(results, features)
        # Should fall back to AnswerGovernor (all constraints allow → TRUSTWORTHY)
        assert isinstance(decision, GovernanceDecision)
        assert decision.mode == AnswerMode.TRUSTWORTHY
