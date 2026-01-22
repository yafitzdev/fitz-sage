# fitz_ai/engines/fitz_rag/retrieval/steps/freshness.py
"""
Freshness Step - Adjust scores based on recency and authority.

Detects query intent (recency/authority) and boosts chunks accordingly:
- Recency: "latest", "recent", "current" → boost newer documents
- Authority: "official", "spec" → boost authoritative sources

Only applies when query signals intent, otherwise passes through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import RetrievalStep

logger = get_logger(__name__)

# Authority scores by source type
AUTHORITY_SCORES = {
    "spec": 1.0,
    "design": 0.8,
    "document": 0.6,
    "notes": 0.4,
}


@dataclass
class FreshnessStep(RetrievalStep):
    """
    Adjust chunk scores based on recency and source authority.

    Only activates when query contains recency or authority keywords.
    Otherwise, chunks pass through unchanged.

    Args:
        recency_weight: How much recency affects score (0-1, default: 0.15)
        authority_weight: How much authority affects score (0-1, default: 0.15)
        recency_half_life_days: Days until recency score halves (default: 90)
    """

    recency_weight: float = 0.15
    authority_weight: float = 0.15
    recency_half_life_days: float = 90.0

    # Keywords that trigger recency boosting
    recency_keywords: list[str] = field(
        default_factory=lambda: [
            "latest",
            "recent",
            "current",
            "new",
            "updated",
            "newest",
            "now",
            "today",
        ]
    )

    # Keywords that trigger authority boosting
    authority_keywords: list[str] = field(
        default_factory=lambda: [
            "official",
            "spec",
            "specification",
            "requirement",
            "authoritative",
            "canonical",
            "standard",
            "definitive",
        ]
    )

    def _has_keyword(self, query_words: set[str], keywords: list[str]) -> bool:
        """Check if any keyword appears as a whole word in query."""
        return bool(query_words & set(keywords))

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Apply freshness/authority adjustments based on query intent."""
        if not chunks:
            return chunks

        # Tokenize query into words for whole-word matching
        # This avoids false positives like "now" in "know"
        import re

        query_words = set(re.findall(r"\b\w+\b", query.lower()))

        # Detect intent from query keywords (whole word match only)
        boost_recency = self._has_keyword(query_words, self.recency_keywords)
        boost_authority = self._has_keyword(query_words, self.authority_keywords)

        if not boost_recency and not boost_authority:
            # No adjustment needed - pass through
            return chunks

        intent = []
        if boost_recency:
            intent.append("recency")
        if boost_authority:
            intent.append("authority")
        logger.debug(f"{RETRIEVER} FreshnessStep: detected intent={intent}")

        now = time()
        adjusted_count = 0

        for chunk in chunks:
            meta = chunk.metadata
            base_score = meta.get("rerank_score") or meta.get("vector_score", 0.5)
            adjustment = 0.0

            # Recency boost
            if boost_recency and self.recency_weight > 0:
                modified_at = meta.get("modified_at")
                if modified_at:
                    age_days = (now - modified_at) / 86400
                    # Exponential decay: score = 0.5^(age/half_life)
                    recency_score = 0.5 ** (age_days / self.recency_half_life_days)
                    adjustment += self.recency_weight * recency_score

            # Authority boost
            if boost_authority and self.authority_weight > 0:
                source_type = meta.get("source_type", "document")
                authority_score = AUTHORITY_SCORES.get(source_type, 0.6)
                adjustment += self.authority_weight * authority_score

            if adjustment > 0:
                # Store adjusted score (additive, capped at 1.0)
                meta["freshness_score"] = min(1.0, base_score + adjustment)
                adjusted_count += 1

        # Re-sort by freshness score (or original score if no freshness)
        def sort_key(c: Chunk) -> float:
            return (
                c.metadata.get("freshness_score")
                or c.metadata.get("rerank_score")
                or c.metadata.get("vector_score", 0)
            )

        chunks.sort(key=sort_key, reverse=True)

        logger.debug(f"{RETRIEVER} FreshnessStep: adjusted {adjusted_count}/{len(chunks)} chunks")

        return chunks
