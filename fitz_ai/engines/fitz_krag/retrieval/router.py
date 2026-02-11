# fitz_ai/engines/fitz_krag/retrieval/router.py
"""
Retrieval router — dispatches queries to available strategies and merges results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fitz_ai.engines.fitz_krag.types import Address

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
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

    def retrieve(self, query: str) -> list[Address]:
        """
        Retrieve addresses from all enabled strategies.

        Runs code + section strategies, merges results by score.
        Chunk fallback supplements when other results are insufficient.
        """
        limit = self._config.top_addresses
        all_addresses: list[Address] = []

        # Run code strategy
        code_addresses = self._code_strategy.retrieve(query, limit)
        all_addresses.extend(code_addresses)

        # Run section strategy if available
        if self._section_strategy:
            section_addresses = self._section_strategy.retrieve(query, limit)
            all_addresses.extend(section_addresses)

        # Optionally run chunk fallback when other results are insufficient
        if (
            self._chunk_strategy
            and self._config.fallback_to_chunks
            and len(all_addresses) < limit // 2
        ):
            chunk_limit = limit - len(all_addresses)
            chunk_addresses = self._chunk_strategy.retrieve(query, chunk_limit)
            all_addresses.extend(chunk_addresses)

        # Merge, deduplicate, and rank by score
        return self._merge_and_rank(all_addresses, limit)

    def _merge_and_rank(
        self,
        addresses: list[Address],
        limit: int,
    ) -> list[Address]:
        """Deduplicate by source_id+location, then rank by score."""
        seen: set[tuple[str, str]] = set()
        merged: list[Address] = []

        for addr in addresses:
            key = (addr.source_id, addr.location)
            if key not in seen:
                seen.add(key)
                merged.append(addr)

        # Sort by score descending
        merged.sort(key=lambda a: a.score, reverse=True)
        return merged[:limit]
