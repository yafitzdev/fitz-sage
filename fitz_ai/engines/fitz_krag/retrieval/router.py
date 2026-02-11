# fitz_ai/engines/fitz_krag/retrieval/router.py
"""
Retrieval router — dispatches queries to available strategies and merges results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.types import Address

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.engines.fitz_krag.query_analyzer import QueryAnalysis
    from fitz_ai.engines.fitz_krag.retrieval.strategies.chunk_fallback import (
        ChunkFallbackStrategy,
    )
    from fitz_ai.engines.fitz_krag.retrieval.strategies.code_search import (
        CodeSearchStrategy,
    )
    from fitz_ai.engines.fitz_krag.retrieval.strategies.section_search import (
        SectionSearchStrategy,
    )

logger = logging.getLogger(__name__)


class RetrievalRouter:
    """Routes queries to available strategies, merges results."""

    def __init__(
        self,
        code_strategy: "CodeSearchStrategy",
        chunk_strategy: "ChunkFallbackStrategy | None",
        config: "FitzKragConfig",
        section_strategy: "SectionSearchStrategy | None" = None,
    ):
        self._code_strategy = code_strategy
        self._chunk_strategy = chunk_strategy
        self._section_strategy = section_strategy
        self._config = config

    def retrieve(
        self,
        query: str,
        analysis: "QueryAnalysis | None" = None,
        detection: "Any | None" = None,
    ) -> list[Address]:
        """
        Retrieve addresses using strategy weights from query analysis.

        When analysis is provided, strategies with near-zero weight are skipped
        and results are ranked using CrossStrategyRanker. Without analysis,
        all strategies run equally (backward compatible).

        When detection is provided (DetectionSummary), it enhances retrieval:
        - Query expansion: additional retrievals with detection.query_variations
        - Comparison: search both detection.comparison_entities
        - Fetch multiplier: increase limit by detection.fetch_multiplier
        """
        from fitz_ai.engines.fitz_krag.retrieval.ranker import CrossStrategyRanker

        limit = self._config.top_addresses

        # Apply fetch multiplier from detection
        if detection and hasattr(detection, "fetch_multiplier"):
            limit = limit * detection.fetch_multiplier

        weights = analysis.strategy_weights if analysis else None
        all_addresses: list[Address] = []

        # Collect queries to run (original + detection expansions)
        queries = [query]
        if detection:
            if hasattr(detection, "query_variations") and detection.query_variations:
                queries.extend(detection.query_variations)
            if hasattr(detection, "comparison_entities") and detection.comparison_entities:
                for entity in detection.comparison_entities:
                    queries.append(f"{query} {entity}")

        for q in queries:
            # Run code strategy (skip if weight below threshold)
            if not weights or weights.get("code", 1.0) > 0.05:
                code_addresses = self._code_strategy.retrieve(q, limit)
                all_addresses.extend(code_addresses)

            # Run section strategy if available and weighted
            if self._section_strategy and (not weights or weights.get("section", 1.0) > 0.05):
                section_addresses = self._section_strategy.retrieve(q, limit)
                all_addresses.extend(section_addresses)

        # Chunk fallback when other results are insufficient
        if (
            self._chunk_strategy
            and self._config.fallback_to_chunks
            and (not weights or weights.get("chunk", 1.0) > 0.05)
            and len(all_addresses) < self._config.top_addresses // 2
        ):
            chunk_limit = self._config.top_addresses - len(all_addresses)
            chunk_addresses = self._chunk_strategy.retrieve(query, chunk_limit)
            all_addresses.extend(chunk_addresses)

        # Deduplicate
        deduped = self._deduplicate(all_addresses)

        # Rank using analysis if available
        if analysis:
            ranker = CrossStrategyRanker()
            ranked = ranker.rank(deduped, analysis)
            return ranked[: self._config.top_addresses]

        # Fallback: sort by score
        deduped.sort(key=lambda a: a.score, reverse=True)
        return deduped[: self._config.top_addresses]

    def _deduplicate(self, addresses: list[Address]) -> list[Address]:
        """Deduplicate addresses by source_id+location."""
        seen: set[tuple[str, str]] = set()
        result: list[Address] = []

        for addr in addresses:
            key = (addr.source_id, addr.location)
            if key not in seen:
                seen.add(key)
                result.append(addr)

        return result
