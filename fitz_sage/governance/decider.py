# fitz_sage/governance/decider.py
"""
ML-based governance decider using a 4-question atomic cascade classifier
with an optional LLM judge for post-classifier correction.

4-question cascade:
  Q1: Is evidence sufficient?      → No = ABSTAIN  (ML)
  Q2: Is there material conflict?  → routes Q3/Q4  (rule: ca_fired)
  Q3: Is the conflict resolved?    → Yes=TRUSTWORTHY, No=DISPUTED (ML)
  Q4: Is evidence truly solid?     → Yes=TRUSTWORTHY, No=ABSTAIN  (ML)

LLM judge (post-classifier):
  Classifier says TRUSTWORTHY → judge says NO  → demote to ABSTAIN
  Classifier says ABSTAIN     → judge says YES → promote to TRUSTWORTHY
  (Disputed is never overridden — dispute detection is evidence-vs-evidence,
   not query-vs-evidence, so the judge can't help there.)

3-class output mapped to AnswerMode:
  abstain → ABSTAIN
  disputed → DISPUTED
  trustworthy → TRUSTWORTHY
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.governance.constraints.feature_extractor import get_feature_type_sets
from fitz_sage.governance.governor import AnswerGovernor, GovernanceDecision

if TYPE_CHECKING:
    from fitz_sage.governance.constraints.base import ConstraintResult
    from fitz_sage.governance.protocol import EvidenceItem
    from fitz_sage.llm.providers.base import ChatProvider

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """Given this question and retrieved evidence, does the evidence DIRECTLY answer the SPECIFIC question asked?

Consider:
- Does it answer THIS EXACT question, or a related but different one?
- Does it cover the specific entity, model, version, or scope asked about?
- Is the data specific enough (not just general category when a specific item is asked)?

Question: {query}
Evidence: {evidence}

Answer only YES or NO, then a one-sentence reason."""

# Feature type sets derived from constraint schemas (single source of truth)
_CATEGORICAL_FEATURES, _BOOL_FEATURES = get_feature_type_sets()

# Default model search paths (relative to package root)
_MODEL_FILENAME = "model_v6_cascade.joblib"


def _find_model_path() -> Path | None:
    """Find the model artifact shipped with the package."""
    path = Path(__file__).resolve().parent / "data" / _MODEL_FILENAME
    return path if path.exists() else None


class GovernanceDecider:
    """ML-based governance decider with fail-open fallback to AnswerGovernor."""

    def __init__(
        self,
        model_path: Path | None = None,
        chat: "ChatProvider | None" = None,
    ) -> None:
        self._available = False
        self._governor = AnswerGovernor()
        self._chat = chat  # For LLM judge (optional)

        path = model_path or _find_model_path()
        if path is None:
            logger.info("GovernanceDecider: no model found, using AnswerGovernor fallback")
            return

        try:
            import joblib

            artifact = joblib.load(path)
            mode = artifact.get("mode", "twostage")

            if mode == "cascade":
                self._mode = "cascade"
                self._q1_model = artifact["q1_model"]
                self._q2_model = artifact.get("q2_model")  # ML router (None in legacy)
                self._q3_model = artifact["q3_model"]
                self._q4_model = artifact["q4_model"]
                self._q1_threshold = artifact.get("q1_threshold", 0.5)
                self._q2_threshold = artifact.get("q2_threshold", 0.5)
                self._q3_threshold = artifact.get("q3_threshold", 0.5)
                self._q4_threshold = artifact.get("q4_threshold", 0.5)
                self._conflict_feature = artifact.get("conflict_feature", "ca_fired")
                self._encoders = artifact.get("encoders", {})
                self._feature_names = artifact["feature_names"]
                self._available = True
                q2_desc = (
                    f"q2={self._q2_threshold:.2f}"
                    if self._q2_model is not None
                    else f"q2=rule({self._conflict_feature})"
                )
                logger.info(
                    f"GovernanceDecider: loaded cascade model from {path} "
                    f"(q1={self._q1_threshold:.2f}, {q2_desc}, "
                    f"q3={self._q3_threshold:.2f}, q4={self._q4_threshold:.2f}, "
                    f"{len(self._feature_names)} features)"
                )
            else:
                self._mode = "twostage"
                self._s1_model = artifact["stage1_model"]
                self._s2_model = artifact["stage2_model"]
                self._s1_threshold = artifact.get("stage1_threshold", 0.5)
                self._s2_threshold = artifact.get("stage2_threshold", 0.5)
                self._encoders = artifact.get("encoders", {})
                self._feature_names = artifact["feature_names"]
                self._available = True
                logger.info(
                    f"GovernanceDecider: loaded twostage model from {path} "
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
        query: str | None = None,
        chunks: Sequence["EvidenceItem"] | None = None,
    ) -> GovernanceDecision:
        """Produce a governance decision using the ML classifier + optional LLM judge.

        The LLM judge corrects the classifier on the trustworthy/abstain boundary:
        - Classifier says TRUSTWORTHY → judge says NO → demote to ABSTAIN
        - Classifier says ABSTAIN → judge says YES → promote to TRUSTWORTHY

        Falls back to AnswerGovernor if the model is unavailable or prediction fails.
        """
        if not self._available or not features:
            return self._governor.decide(constraint_results)

        try:
            label, confidence = self._predict(features)
            logger.debug(
                f"GovernanceDecider: ML prediction={label} (conf={confidence:.3f}) | "
                f"ie_signal={features.get('ie_signal')} "
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

        # LLM judge: correct trustworthy/abstain boundary
        if self._chat and query and chunks:
            mode = self._apply_llm_judge(mode, query, chunks)

        return GovernanceDecision(
            mode=mode,
            triggered_constraints=tuple(triggered),
            reasons=tuple(reasons),
            signals=frozenset(signals),
        )

    def _apply_llm_judge(
        self,
        mode: AnswerMode,
        query: str,
        chunks: Sequence["EvidenceItem"],
    ) -> AnswerMode:
        """Apply LLM judge to correct trustworthy/abstain boundary.

        - TRUSTWORTHY + judge says NO → ABSTAIN (catch false-trustworthy)
        - ABSTAIN + judge says YES → TRUSTWORTHY (recover false-abstain)
        - DISPUTED is never overridden (dispute is evidence-vs-evidence)
        """
        if mode == AnswerMode.DISPUTED:
            return mode

        if not chunks:
            return mode

        evidence = chunks[0].content[:500]
        try:
            response = self._chat.chat(
                [{"role": "user", "content": _JUDGE_PROMPT.format(query=query, evidence=evidence)}],
                temperature=0.0,
                max_tokens=50,
            )
            word = response.strip().upper().split()[0] if response.strip() else ""
            judge_says_yes = word.startswith("YES")
        except Exception as e:
            logger.warning(f"GovernanceDecider: LLM judge failed, keeping ML prediction: {e}")
            return mode

        if mode == AnswerMode.TRUSTWORTHY and not judge_says_yes:
            logger.info(
                "GovernanceDecider: LLM judge demoted TRUSTWORTHY → ABSTAIN "
                "(evidence does not directly answer the specific question)"
            )
            return AnswerMode.ABSTAIN

        if mode == AnswerMode.ABSTAIN and judge_says_yes:
            logger.info(
                "GovernanceDecider: LLM judge promoted ABSTAIN → TRUSTWORTHY "
                "(evidence directly answers the question despite classifier doubt)"
            )
            return AnswerMode.TRUSTWORTHY

        return mode

    def _predict(self, features: dict[str, Any]) -> tuple[str, float]:
        """Run prediction. Dispatches to cascade or twostage based on model type."""
        if self._mode == "cascade":
            return self._predict_cascade(features)
        return self._predict_twostage(features)

    def _predict_cascade(self, features: dict[str, Any]) -> tuple[str, float]:
        """Run 4-question cascade prediction. Returns (label, confidence).

        Q1: Is evidence sufficient? → No = abstain
        Q2: Is there material conflict? (rule: ca_fired)
        Q3: conflict path → Is conflict resolved? → Yes=trustworthy, No=disputed
        Q4: clean path → Is evidence solid? → Yes=trustworthy, No=abstain
        """
        X = self._prepare_features(features)

        # Q1: Is evidence sufficient?
        q1_probas = self._q1_model.predict_proba(X)
        q1_classes = list(self._q1_model.classes_)
        try:
            answerable_idx = q1_classes.index(1)
        except ValueError:
            answerable_idx = q1_classes.index("answerable")
        p_answerable = float(q1_probas[0, answerable_idx])

        if p_answerable < self._q1_threshold:
            return "abstain", 1.0 - p_answerable

        # Q2: Is there material conflict? (ML router or legacy rule fallback)
        if self._q2_model is not None:
            q2_probas = self._q2_model.predict_proba(X)
            q2_classes = list(self._q2_model.classes_)
            try:
                conflict_idx = q2_classes.index(1)
            except ValueError:
                conflict_idx = q2_classes.index("conflict")
            has_conflict = float(q2_probas[0, conflict_idx]) >= self._q2_threshold
        else:
            # Legacy fallback: hard rule on ca_fired
            has_conflict = bool(features.get(self._conflict_feature, False))

        if has_conflict:
            # Q3: Is the conflict resolved?
            q3_probas = self._q3_model.predict_proba(X)
            q3_classes = list(self._q3_model.classes_)
            try:
                resolved_idx = q3_classes.index(1)
            except ValueError:
                resolved_idx = q3_classes.index("resolved")
            p_resolved = float(q3_probas[0, resolved_idx])

            if p_resolved >= self._q3_threshold:
                return "trustworthy", p_resolved
            else:
                return "disputed", 1.0 - p_resolved
        else:
            # Q4: Is evidence truly solid?
            q4_probas = self._q4_model.predict_proba(X)
            q4_classes = list(self._q4_model.classes_)
            try:
                solid_idx = q4_classes.index(1)
            except ValueError:
                solid_idx = q4_classes.index("solid")
            p_solid = float(q4_probas[0, solid_idx])

            if p_solid >= self._q4_threshold:
                return "trustworthy", p_solid
            else:
                return "abstain", 1.0 - p_solid

    def _predict_twostage(self, features: dict[str, Any]) -> tuple[str, float]:
        """Run two-stage prediction (legacy). Returns (label, confidence)."""
        X = self._prepare_features(features)

        # Stage 1: answerable vs abstain
        s1_probas = self._s1_model.predict_proba(X)
        s1_classes = list(self._s1_model.classes_)
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
        import warnings

        import pandas as pd

        # Suppress PerformanceWarning from repeated column inserts on a
        # single-row DataFrame — not a real performance issue at this scale.
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

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

        # Log-transform ctx_number_variance (matches training)
        if "ctx_number_variance" in X.columns:
            import numpy as np

            X["ctx_number_variance"] = np.log1p(X["ctx_number_variance"])

        # Interaction features (must match prepare_features in train_classifier.py)
        if "ca_fired" in X.columns and "mean_vector_score" in X.columns:
            X["ix_ca_x_vector"] = X["ca_fired"] * X["mean_vector_score"]
        if "ctx_contradiction_count" in X.columns and "ctx_total_chars" in X.columns:
            X["ix_contradiction_density"] = X["ctx_contradiction_count"] / (
                X["ctx_total_chars"] + 1
            )
        if "ctx_negation_count" in X.columns and "ctx_total_chars" in X.columns:
            X["ix_negation_density"] = X["ctx_negation_count"] / (X["ctx_total_chars"] + 1)
        if "mean_vector_score" in X.columns and "score_spread" in X.columns:
            X["ix_score_confidence"] = X["mean_vector_score"] - X["score_spread"]
        if "ca_pairs_checked" in X.columns and "has_disputed_signal" in X.columns:
            X["ix_ca_pairs_x_disputed"] = X["ca_pairs_checked"] * X["has_disputed_signal"]
        if "ie_max_similarity" in X.columns and "ie_fired" in X.columns:
            X["ix_ie_sim_x_fired"] = X["ie_max_similarity"] * X["ie_fired"]
        if "av_citation_quality" in X.columns and "mean_vector_score" in X.columns:
            X["ix_citation_x_vector"] = X["av_citation_quality"] * X["mean_vector_score"]
        if "has_disputed_signal" in X.columns and "av_citation_found" in X.columns:
            X["ix_disputed_x_av"] = X["has_disputed_signal"] * (1 - X["av_citation_found"])
        if "ctx_max_pairwise_sim" in X.columns and "ctx_min_pairwise_sim" in X.columns:
            X["ix_ctx_sim_spread"] = X["ctx_max_pairwise_sim"] - X["ctx_min_pairwise_sim"]
        if "mean_vector_score" in X.columns and "ctx_contradiction_count" in X.columns:
            X["ix_vector_x_contradiction"] = X["mean_vector_score"] * X["ctx_contradiction_count"]
        if "av_fired" in X.columns and "ie_fired" in X.columns:
            X["ix_av_no_ie"] = X["av_fired"] * (1 - X["ie_fired"])
        if "av_citation_found" in X.columns and "ie_fired" in X.columns:
            X["ix_av_nocite_no_ie"] = (1 - X["av_citation_found"]) * (1 - X["ie_fired"])
        if "num_constraints_fired" in X.columns:
            X["ix_multi_denial"] = (X["num_constraints_fired"] >= 2).astype(int)
        if "has_any_denial" in X.columns and "mean_vector_score" in X.columns:
            X["ix_denial_low_vector"] = X["has_any_denial"] * (1 - X["mean_vector_score"])
        if "av_citation_found" in X.columns and "has_abstain_signal" in X.columns:
            X["ix_av_x_abstain"] = (1 - X["av_citation_found"]) * X["has_abstain_signal"]
        if "has_cross_chunk_divergence" in X.columns and "ca_fired" in X.columns:
            X["ix_divergence_no_ca"] = X["has_cross_chunk_divergence"] * (1 - X["ca_fired"])
        if "has_within_chunk_divergence" in X.columns and "ca_fired" in X.columns:
            X["ix_any_divergence_no_ca"] = (
                X.get("has_cross_chunk_divergence", 0) | X["has_within_chunk_divergence"]
            ) * (1 - X["ca_fired"])

        # Q3-specific features: distinguish data-rich trustworthy from genuine disputes
        if "ctx_number_count" in X.columns and "num_chunks" in X.columns:
            X["numerical_richness_per_chunk"] = X["ctx_number_count"] / X["num_chunks"].clip(
                lower=1
            )
        if "within_chunk_num_conflicts" in X.columns and "cross_chunk_num_conflicts" in X.columns:
            total_conf = X["cross_chunk_num_conflicts"] + X["within_chunk_num_conflicts"]
            X["conflict_internality_ratio"] = X["within_chunk_num_conflicts"] / total_conf.clip(
                lower=1
            )
        if "ctx_total_chars" in X.columns and "num_chunks" in X.columns:
            X["chars_per_chunk"] = X["ctx_total_chars"] / X["num_chunks"].clip(lower=1)
        if "ctx_contradiction_count" in X.columns and "ctx_total_chars" in X.columns:
            X["contradiction_per_char"] = X["ctx_contradiction_count"] / X["ctx_total_chars"].clip(
                lower=1
            )

        # Conflict quality features (Q3/FT-disputed)
        if "cross_chunk_num_conflicts" in X.columns and "ctx_number_count" in X.columns:
            X["conflict_to_number_ratio"] = X["cross_chunk_num_conflicts"] / X[
                "ctx_number_count"
            ].clip(lower=1)
        if "ctx_negation_count" in X.columns and "ctx_total_chars" in X.columns:
            X["negation_per_char"] = X["ctx_negation_count"] / X["ctx_total_chars"].clip(lower=1)

        # Q1 recovery: partial-answer detection
        if "ctx_total_chars" in X.columns and "vocab_overlap_ratio" in X.columns:
            X["short_ctx_with_overlap"] = (
                (X["ctx_total_chars"] < 500) & (X["vocab_overlap_ratio"] > 0.3)
            ).astype(int)

        # New interaction features
        if "av_fired" in X.columns and "vocab_overlap_ratio" in X.columns:
            X["ix_av_fires_good_overlap"] = X["av_fired"] * X["vocab_overlap_ratio"]
        if "cross_chunk_max_divergence" in X.columns and "cross_chunk_num_conflicts" in X.columns:
            X["ix_max_div_per_conflict"] = X["cross_chunk_max_divergence"] / (
                X["cross_chunk_num_conflicts"].clip(lower=1)
            )
        if "num_chunks" in X.columns and "has_any_denial" in X.columns:
            X["ix_single_chunk_denial"] = (
                (X["num_chunks"] == 1) & (X["has_any_denial"] == 1)
            ).astype(int)
        if "ie_fired" in X.columns and "ca_fired" in X.columns:
            X["ix_ie_no_ca"] = X["ie_fired"] * (1 - X["ca_fired"])
            X["ix_ca_no_ie"] = X["ca_fired"] * (1 - X["ie_fired"])

        # Q2 routing: number-rich docs are not disputes
        if "number_density" in X.columns and "conflict_to_number_ratio" in X.columns:
            X["ix_numrich_low_conflict"] = X["number_density"] * (1 - X["conflict_to_number_ratio"])
        # Q1/Q4 recovery: no constraints + good signals = answerable
        if "num_constraints_fired" in X.columns and "query_subject_partial" in X.columns:
            X["ix_no_constraint_good_signal"] = (
                (X["num_constraints_fired"] == 0).astype(float)
                * X.get("query_subject_partial", 0)
                * X.get("mean_vector_score", 0)
            )
        # Hedged disputes: hedged evidence + divergence = dispute not abstain
        if "assertion_density" in X.columns and "has_cross_chunk_divergence" in X.columns:
            X["ix_hedged_with_conflicts"] = (1 - X["assertion_density"]) * X[
                "has_cross_chunk_divergence"
            ]

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
