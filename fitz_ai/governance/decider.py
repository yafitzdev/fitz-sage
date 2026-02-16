# fitz_ai/governance/decider.py
"""
ML-based governance decider using a two-stage calibrated classifier.

Replaces hand-coded AnswerGovernor priority rules with a trained classifier
that predicts governance mode from constraint features. Falls back to
AnswerGovernor on any error (fail-open).

Two-stage prediction:
  Stage 1: answerable vs abstain (RF)
  Stage 2: trustworthy vs disputed (ET), only for answerable cases

3-class output mapped to AnswerMode:
  abstain → ABSTAIN
  disputed → DISPUTED
  trustworthy → TRUSTWORTHY
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.governance.governor import AnswerGovernor, GovernanceDecision

if TYPE_CHECKING:
    from fitz_ai.governance.constraints.base import ConstraintResult

logger = logging.getLogger(__name__)

# Feature type sets (must match train_classifier.py exactly)
_CATEGORICAL_FEATURES = {
    "ie_signal",
    "ca_signal",
    "ca_first_evidence_char",
    "ca_evidence_characters",
    "caa_query_type",
    "sit_info_type_requested",
}

_BOOL_FEATURES = {
    "ie_fired",
    "ca_fired",
    "ca_numerical_variance_detected",
    "caa_fired",
    "caa_has_causal_evidence",
    "caa_has_predictive_evidence",
    "sit_fired",
    "sit_entity_mismatch",
    "sit_has_specific_info",
    "has_qualified_signal",
    "detection_temporal",
    "detection_comparison",
    "has_distinct_years",
}

# Default model search paths (relative to package root)
_MODEL_FILENAME = "model_v5_calibrated.joblib"


def _find_model_path() -> Path | None:
    """Search for the model artifact in known locations."""
    candidates = [
        # Development: tools/governance/data/ (project root)
        Path(__file__).resolve().parent.parent.parent
        / "tools"
        / "governance"
        / "data"
        / _MODEL_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


class GovernanceDecider:
    """ML-based governance decider with fail-open fallback to AnswerGovernor."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._available = False
        self._governor = AnswerGovernor()

        path = model_path or _find_model_path()
        if path is None:
            logger.info("GovernanceDecider: no model found, using AnswerGovernor fallback")
            return

        try:
            import joblib

            artifact = joblib.load(path)
            self._s1_model = artifact["stage1_model"]
            self._s2_model = artifact["stage2_model"]
            self._s1_threshold = artifact.get("stage1_threshold", 0.5)
            self._s2_threshold = artifact.get("stage2_threshold", 0.5)
            self._encoders = artifact.get("encoders", {})
            self._feature_names = artifact["feature_names"]
            self._available = True
            logger.info(
                f"GovernanceDecider: loaded model from {path} "
                f"(s1={self._s1_threshold:.2f}, s2={self._s2_threshold:.2f}, "
                f"{len(self._feature_names)} features)"
            )
        except Exception as e:
            logger.warning(f"GovernanceDecider: failed to load model from {path}: {e}")

    @property
    def available(self) -> bool:
        """Whether the ML model is loaded and available."""
        return self._available

    def decide(
        self,
        constraint_results: Sequence[ConstraintResult],
        features: dict[str, Any] | None = None,
    ) -> GovernanceDecision:
        """Produce a governance decision using the ML classifier.

        Falls back to AnswerGovernor if the model is unavailable or prediction fails.
        """
        if not self._available or not features:
            return self._governor.decide(constraint_results)

        try:
            label, confidence = self._predict(features)
            logger.debug(
                f"GovernanceDecider: ML prediction={label} (conf={confidence:.3f}) | "
                f"ie_signal={features.get('ie_signal')} ie_sim={features.get('ie_max_similarity')} "
                f"chunks={features.get('num_chunks')} denials={features.get('num_constraints_fired')} "
                f"vocab_overlap={features.get('vocab_overlap_ratio')}"
            )
        except Exception as e:
            logger.warning(f"GovernanceDecider: prediction failed, falling back: {e}")
            return self._governor.decide(constraint_results)

        # Collect constraint metadata for GovernanceDecision fields
        triggered, reasons, signals = self._collect_constraint_info(constraint_results)

        # Map 3-class label to AnswerMode
        mode = self._map_to_answer_mode(label, triggered)

        return GovernanceDecision(
            mode=mode,
            triggered_constraints=tuple(triggered),
            reasons=tuple(reasons),
            signals=frozenset(signals),
        )

    def _predict(self, features: dict[str, Any]) -> tuple[str, float]:
        """Run two-stage prediction. Returns (label, confidence).

        Stage 1 classes: 0=abstain, 1=answerable (int labels from training).
        Stage 2 classes: 0=disputed, 1=trustworthy (int labels from training).
        """
        X = self._prepare_features(features)

        # Stage 1: answerable vs abstain
        s1_probas = self._s1_model.predict_proba(X)
        s1_classes = list(self._s1_model.classes_)
        # Class 1 = answerable (may be int or string depending on model)
        try:
            answerable_idx = s1_classes.index(1)
        except ValueError:
            answerable_idx = s1_classes.index("answerable")
        p_answerable = float(s1_probas[0, answerable_idx])

        if p_answerable < self._s1_threshold:
            return "abstain", 1.0 - p_answerable

        # Stage 2: trustworthy vs disputed
        s2_probas = self._s2_model.predict_proba(X)
        s2_classes = list(self._s2_model.classes_)
        # Class 1 = trustworthy (may be int or string depending on model)
        try:
            trustworthy_idx = s2_classes.index(1)
        except ValueError:
            trustworthy_idx = s2_classes.index("trustworthy")
        p_trustworthy = float(s2_probas[0, trustworthy_idx])

        if p_trustworthy >= self._s2_threshold:
            return "trustworthy", p_trustworthy
        else:
            return "disputed", 1.0 - p_trustworthy

    def _prepare_features(self, features: dict[str, Any]):
        """Convert raw feature dict to a model-ready single-row DataFrame."""
        import pandas as pd

        X = pd.DataFrame([features])

        # Encode categoricals with saved LabelEncoders
        for col in _CATEGORICAL_FEATURES:
            if col not in X.columns:
                X[col] = "none"
            X[col] = X[col].fillna("none").astype(str)
            if col in self._encoders:
                le = self._encoders[col]
                known = set(le.classes_)
                X[col] = X[col].apply(
                    lambda v, k=known, e=le: e.transform([v])[0] if v in k else -1
                )
            else:
                X[col] = 0

        # Convert bools to int
        for col in _BOOL_FEATURES:
            if col not in X.columns:
                X[col] = 0
            else:
                X[col] = (
                    X[col].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0).astype(int)
                )

        # Fill NaN, convert all to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        # Align columns to model's expected feature set
        missing = [c for c in self._feature_names if c not in X.columns]
        for col in missing:
            X[col] = 0
        X = X[self._feature_names]

        return X

    @staticmethod
    def _collect_constraint_info(
        constraint_results: Sequence[ConstraintResult],
    ) -> tuple[list[str], list[str], set[str]]:
        """Extract triggered constraints, reasons, and signals."""
        triggered: list[str] = []
        reasons: list[str] = []
        signals: set[str] = set()

        for result in constraint_results:
            if not result.allow_decisive_answer:
                name = result.metadata.get("constraint_name", "unknown")
                triggered.append(name)
                if result.reason:
                    reasons.append(result.reason)
                if result.signal:
                    signals.add(result.signal)

        return triggered, reasons, signals

    @staticmethod
    def _map_to_answer_mode(label: str, triggered: list[str]) -> AnswerMode:
        """Map 3-class prediction to AnswerMode."""
        if label == "abstain":
            return AnswerMode.ABSTAIN
        if label == "disputed":
            return AnswerMode.DISPUTED
        return AnswerMode.TRUSTWORTHY


__all__ = ["GovernanceDecider"]
