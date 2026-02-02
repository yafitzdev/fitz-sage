# fitz_ai/core/guardrails/plugins/insufficient_evidence.py
"""
Insufficient Evidence Constraint - Default guardrail for evidence coverage.

This constraint prevents the system from giving confident answers when
there is no relevant evidence in the retrieved chunks.

Three rules (checked in order):
1. Empty context = ABSTAIN
2. If vector_score available and all < 0.3 = ABSTAIN
3. If no vector_score, check lexical overlap (zero shared words = ABSTAIN)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult

logger = get_logger(__name__)

# Below this score, vectors are nearly orthogonal (no semantic relationship)
MIN_RELEVANCE_SCORE = 0.3

# Common stopwords to ignore in overlap check
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "and", "but",
    "if", "or", "because", "until", "while", "about", "what", "which",
    "who", "whom", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours",
    "theirs", "any", "both", "either", "neither", "much", "many",
})


def _extract_words(text: str) -> set[str]:
    """Extract meaningful words (lowercase, no stopwords, min 3 chars)."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _get_max_score(chunks: Sequence[Chunk]) -> float | None:
    """Get the highest vector_score from chunks, or None if no scores."""
    scores = []
    for chunk in chunks:
        score = chunk.metadata.get("vector_score")
        if score is not None:
            scores.append(float(score))
    return max(scores) if scores else None


def _has_lexical_overlap(query: str, chunks: Sequence[Chunk]) -> bool:
    """Check if query shares any meaningful words with chunks."""
    query_words = _extract_words(query)
    if not query_words:
        return True  # Can't determine overlap, allow

    for chunk in chunks:
        chunk_words = _extract_words(chunk.content)
        if query_words & chunk_words:  # Intersection
            return True

    return False


@dataclass
class InsufficientEvidenceConstraint:
    """
    Constraint that prevents confident answers without relevant evidence.

    Three rules (checked in order):
    1. If no chunks retrieved, abstain
    2. If best chunk score < 0.3, abstain (off-topic retrieval)
    3. If no scores available, check lexical overlap (zero = abstain)

    Attributes:
        enabled: Whether this constraint is active (default: True)
        min_score: Minimum vector score to consider relevant (default: 0.3)
    """

    enabled: bool = True
    min_score: float = MIN_RELEVANCE_SCORE
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def name(self) -> str:
        return "insufficient_evidence"

    def apply(
        self,
        query: str,
        chunks: Sequence[Chunk],
    ) -> ConstraintResult:
        """
        Check if there is relevant evidence to answer the query.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies if no chunks or evidence is off-topic
        """
        if not self.enabled:
            return ConstraintResult.allow()

        # Rule 1: No chunks at all
        if not chunks:
            logger.info(f"{PIPELINE} InsufficientEvidenceConstraint: no chunks retrieved")
            return ConstraintResult.deny(
                reason="No evidence retrieved",
                signal="abstain",
                evidence_count=0,
            )

        # Rule 2: Check vector_score if available
        max_score = _get_max_score(chunks)
        if max_score is not None:
            if max_score < self.min_score:
                logger.info(
                    f"{PIPELINE} InsufficientEvidenceConstraint: score {max_score:.3f} "
                    f"< {self.min_score} -> ABSTAIN"
                )
                return ConstraintResult.deny(
                    reason=f"Retrieved content not relevant (score={max_score:.3f})",
                    signal="abstain",
                    evidence_count=len(chunks),
                    max_score=max_score,
                )
            # Score is good enough
            return ConstraintResult.allow()

        # Rule 3: No vector_score, use lexical overlap as fallback
        if not _has_lexical_overlap(query, chunks):
            logger.info(
                f"{PIPELINE} InsufficientEvidenceConstraint: no lexical overlap -> ABSTAIN"
            )
            return ConstraintResult.deny(
                reason="Context does not appear related to query",
                signal="abstain",
                evidence_count=len(chunks),
            )

        logger.debug(
            f"{PIPELINE} InsufficientEvidenceConstraint: allowing (lexical overlap found)"
        )
        return ConstraintResult.allow()


__all__ = ["InsufficientEvidenceConstraint"]
