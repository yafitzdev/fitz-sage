# fitz_ai/retrieval/detection/classifier.py
"""
ML + keyword-based query classifier for detection routing.

Loads a pre-trained joblib artifact and applies per-label thresholds
to gate which LLM detection modules are needed for a given query.
Keyword rules handle ungated categories (aggregation, freshness, rewriter)
without requiring ML predictions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from fitz_ai.logging.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
_MODEL_PATH = _DATA_DIR / "model_v1_detection.joblib"

# Keyword patterns for ungated categories
_AGGREGATION_RE = re.compile(
    r"\b(list all|list every|show all|how many|count of|enumerate|what are all|every)\b",
    re.IGNORECASE,
)
_FRESHNESS_RE = re.compile(
    r"\b(latest|most recent|current|right now|today|newest|recently|new version)\b",
    re.IGNORECASE,
)
_REWRITER_RE = re.compile(
    r"\b(it|they|this|that|these|those|he|she|the previous|the above)\b",
    re.IGNORECASE,
)


class DetectionClassifier:
    """
    ML + keyword detection classifier.

    Uses a pre-trained logistic regression model for temporal/comparison
    detection and keyword regex rules for aggregation, freshness, and rewriter.

    When available, predicts which DetectionCategory labels need LLM processing,
    allowing the orchestrator to skip unnecessary LLM module calls.

    If the model artifact is missing or fails to load, the classifier marks
    itself unavailable and the orchestrator falls back to running all LLM modules.
    """

    def __init__(self) -> None:
        self._model = None
        self._vectorizer = None
        self._labels: list[str] = []
        self._thresholds: dict[str, float] = {}
        self._available = False

        try:
            import joblib

            artifact = joblib.load(_MODEL_PATH)
            self._model = artifact["model"]
            self._vectorizer = artifact["vectorizer"]
            self._labels = artifact["labels"]
            self._thresholds = artifact["thresholds"]
            self._available = True
            logger.debug(
                f"DetectionClassifier loaded from {_MODEL_PATH} "
                f"(labels={self._labels}, thresholds={self._thresholds})"
            )
        except FileNotFoundError:
            logger.debug(
                f"DetectionClassifier model not found at {_MODEL_PATH}; "
                "running in unavailable mode (all LLM modules will run)"
            )
        except Exception as exc:
            logger.warning(f"DetectionClassifier failed to load: {exc}; running in unavailable mode")

    @property
    def available(self) -> bool:
        """True if the model artifact was loaded successfully."""
        return self._available

    def predict(self, query: str) -> set | None:
        """
        Predict which DetectionCategory values need LLM processing.

        Returns None if the classifier is unavailable (caller should run all modules).
        Returns an empty set if no categories need LLM processing.
        Returns a non-empty set of DetectionCategory values that need LLM modules.

        Keyword-gated categories (AGGREGATION, FRESHNESS, REWRITER) are always
        evaluated via regex regardless of ML availability.

        Args:
            query: Raw query string from the user.

        Returns:
            set[DetectionCategory] | None
        """
        from .protocol import DetectionCategory

        if not self._available:
            return None

        try:
            flagged: set = set()

            # ML predictions for temporal and comparison
            vec = self._vectorizer.transform([query])
            # predict_proba returns shape (n_samples, n_classes) per label
            probas = self._model.predict_proba(vec)

            _label_to_category = {
                "temporal": DetectionCategory.TEMPORAL,
                "comparison": DetectionCategory.COMPARISON,
            }

            for idx, label in enumerate(self._labels):
                # probas[idx] is (n_samples, 2); column 1 is P(positive)
                prob_positive = probas[idx][0][1]
                threshold = self._thresholds.get(label, 0.5)
                if prob_positive >= threshold:
                    category = _label_to_category.get(label)
                    if category is not None:
                        flagged.add(category)

            # Keyword rules for ungated categories
            if _AGGREGATION_RE.search(query):
                flagged.add(DetectionCategory.AGGREGATION)
            if _FRESHNESS_RE.search(query):
                flagged.add(DetectionCategory.FRESHNESS)
            if _REWRITER_RE.search(query):
                flagged.add(DetectionCategory.REWRITER)

            return flagged

        except Exception as exc:
            logger.warning(f"DetectionClassifier.predict failed: {exc}; returning None (fail-open)")
            return None


__all__ = ["DetectionClassifier"]
