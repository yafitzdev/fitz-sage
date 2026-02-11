# fitz_ai/engines/fitz_krag/retrieval/ranker.py
"""
Cross-strategy ranker — scores and ranks addresses from multiple strategies.

Uses query analysis weights, entity matching bonuses, and type mismatch
penalties to produce a coherent ranking across code and document results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fitz_ai.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType

logger = logging.getLogger(__name__)

# Bonus for entity name match in address summary/location
ENTITY_MATCH_BONUS = 0.15

# Strategy kind mapping
_KIND_TO_STRATEGY: dict[AddressKind, str] = {
    AddressKind.SYMBOL: "code",
    AddressKind.FILE: "code",
    AddressKind.SECTION: "section",
    AddressKind.CHUNK: "chunk",
}


class CrossStrategyRanker:
    """Ranks addresses across strategies using query analysis."""

    def rank(
        self,
        addresses: list[Address],
        analysis: "QueryAnalysis",
    ) -> list[Address]:
        """
        Score and rank addresses using query analysis weights.

        Scoring:
        1. Base score from strategy's internal ranking (addr.score)
        2. Strategy weight multiplier from query analysis
        3. Entity match bonus when query mentions symbol/section name
        """
        weights = analysis.strategy_weights
        entities = set(e.lower() for e in analysis.entities)

        scored: list[tuple[float, Address]] = []
        for addr in addresses:
            score = self._compute_score(addr, weights, entities)
            scored.append((score, addr))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [addr for _, addr in scored]

    def _compute_score(
        self,
        addr: Address,
        weights: dict[str, float],
        entities: set[str],
    ) -> float:
        """Compute weighted score for an address."""
        # Base score from strategy
        base_score = addr.score

        # Strategy weight multiplier
        strategy = _KIND_TO_STRATEGY.get(addr.kind, "chunk")
        weight = weights.get(strategy, 0.1)
        weighted_score = base_score * weight

        # Entity match bonus
        if entities:
            location_lower = addr.location.lower()
            summary_lower = (addr.summary or "").lower()
            for entity in entities:
                if entity in location_lower or entity in summary_lower:
                    weighted_score += ENTITY_MATCH_BONUS
                    break

        return weighted_score
