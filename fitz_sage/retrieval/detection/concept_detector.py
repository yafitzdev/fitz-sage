# fitz_sage/retrieval/detection/concept_detector.py
"""
Embedding-based concept centroid detector for detection routing.

Uses cosine similarity against concept-phrase centroids to gate which
LLM detection modules are needed for a given query. Semantically captures
temporal, comparison, aggregation, and freshness intent without brittle
regex or bag-of-words TF-IDF.

Pattern mirrors SemanticMatcher in governance/constraints/semantic.py.
Concepts are embedded once and cached; the gate fires when max similarity
to any concept in the cluster exceeds a per-label threshold.

When an embedder is available, ConceptDetector supersedes DetectionClassifier
(TF-IDF) in DetectionOrchestrator. Both return the same set[DetectionCategory] | None
interface so the orchestrator gate logic is unchanged.
"""

from __future__ import annotations

import re
from typing import Callable

from fitz_sage.core.math import cosine_similarity
from fitz_sage.logging.logger import get_logger

logger = get_logger(__name__)

# Type alias — identical to SemanticMatcher
EmbedderFunc = Callable[[str], list[float]]

# ---------------------------------------------------------------------------
# Concept anchor phrases
# ---------------------------------------------------------------------------
# Each tuple defines a semantic cluster. The embedder maps these to vectors;
# at query time the query vector is compared to each concept individually
# and the max similarity is used (high recall > centroid averaging).

TEMPORAL_CONCEPTS: tuple[str, ...] = (
    "when did this happen",
    "what is the timeline of events",
    "how did this change over time",
    "what year did this occur",
    "historical sequence of events",
    "before and after this event",
    "trend over the past years",
    "from when until when",
    "at what point in time",
    "the history of this",
)

COMPARISON_CONCEPTS: tuple[str, ...] = (
    "compare these two options",
    "what is the difference between them",
    "which one is better",
    "pros and cons of each",
    "advantages versus disadvantages",
    "how do they differ from each other",
    "contrast these alternatives",
    "which is superior",
    "similarities and differences between",
    "side by side comparison of",
)

AGGREGATION_CONCEPTS: tuple[str, ...] = (
    "list all the items",
    "show me everything",
    "how many are there in total",
    "enumerate all options",
    "give me a complete list",
    "count the total number of occurrences",
    "what are all the cases",
    "all of the results",
    "total across everything",
    "every single one",
)

FRESHNESS_CONCEPTS: tuple[str, ...] = (
    # Recency sub-signal
    "what is the current status",
    "what is the most recent update",
    "latest version available now",
    "current state right now",
    "most up to date information",
    "what is happening currently",
    "newest information available today",
    "as of today what is",
    "the latest news on this",
    "what changed most recently",
    # Authority sub-signal (maps to boost_authority in FreshnessModule)
    "what is the official recommendation",
    "what is the best practice for this",
    "what is the standard approach",
    "recommended way to do this",
    "proper method according to guidelines",
    "authoritative source on this topic",
    "what does the specification say",
    "canonical approach in the industry",
)

# Keyword fallback for REWRITER — anaphoric pronouns are better handled by
# regex than embeddings (embedding models collapse "it" into generic space)
_REWRITER_RE = re.compile(
    r"\b(it|they|this|that|these|those|he|she|the previous|the above)\b",
    re.IGNORECASE,
)

# Default thresholds: conservative → high recall (gate must not miss true positives)
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "temporal": 0.60,
    "comparison": 0.62,
    "aggregation": 0.60,
    "freshness": 0.60,
}

_CONCEPT_MAP: dict[str, tuple[str, ...]] = {
    "temporal": TEMPORAL_CONCEPTS,
    "comparison": COMPARISON_CONCEPTS,
    "aggregation": AGGREGATION_CONCEPTS,
    "freshness": FRESHNESS_CONCEPTS,
}


class ConceptDetector:
    """
    Embedding-based concept centroid detector for detection routing.

    Mirrors the DetectionClassifier.predict() interface — returns the same
    set[DetectionCategory] | None type — so the orchestrator gate logic is
    a transparent swap: ConceptDetector when embedder available, else TF-IDF.

    Individual concept similarities are used (max over cluster) rather than
    centroid similarity, favouring recall over precision in the gate role.

    Args:
        embedder: Function mapping text → float vector (same as EmbedderFunc).
        thresholds: Per-label thresholds. Defaults to _DEFAULT_THRESHOLDS.
    """

    def __init__(
        self,
        embedder: EmbedderFunc,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._embedder = embedder
        self._thresholds = thresholds or dict(_DEFAULT_THRESHOLDS)
        # Cache: label → list of embedded concept vectors
        self._concept_cache: dict[str, list[list[float]]] = {}

    def _get_concept_vectors(self, label: str) -> list[list[float]]:
        """Return cached concept embeddings for a label, computing on first call."""
        if label not in self._concept_cache:
            concepts = _CONCEPT_MAP[label]
            self._concept_cache[label] = [self._embedder(c) for c in concepts]
        return self._concept_cache[label]

    def _max_similarity(self, query_vec: list[float], label: str) -> float:
        """Return max cosine similarity between query_vec and any concept in label's cluster."""
        concept_vecs = self._get_concept_vectors(label)
        return max(cosine_similarity(query_vec, cv) for cv in concept_vecs)

    def predict(self, query: str) -> set | None:
        """
        Predict which DetectionCategory values need LLM processing.

        Returns None on embedding error (fail-open — caller runs all modules).
        Returns empty set when no categories need LLM processing.
        Returns non-empty set of DetectionCategory values to run.

        Args:
            query: Raw query string from the user.

        Returns:
            set[DetectionCategory] | None
        """
        from .protocol import DetectionCategory

        try:
            query_vec = self._embedder(query)
            flagged: set = set()

            _label_to_category = {
                "temporal": DetectionCategory.TEMPORAL,
                "comparison": DetectionCategory.COMPARISON,
                "aggregation": DetectionCategory.AGGREGATION,
                "freshness": DetectionCategory.FRESHNESS,
            }

            for label, category in _label_to_category.items():
                sim = self._max_similarity(query_vec, label)
                threshold = self._thresholds.get(label, 0.60)
                if sim >= threshold:
                    flagged.add(category)

            # REWRITER: keyword-only (anaphoric pronouns)
            if _REWRITER_RE.search(query):
                flagged.add(DetectionCategory.REWRITER)

            return flagged

        except Exception as exc:
            logger.warning(f"ConceptDetector.predict failed: {exc}; returning None (fail-open)")
            return None


__all__ = ["ConceptDetector", "EmbedderFunc"]
