# fitz_ai/engines/fitz_krag/retrieval/strategies/chunk_fallback.py
"""
Fallback to existing pgvector chunk search for non-code queries.

Wraps chunk search results as Address(kind=CHUNK) objects so they can
be processed by the same reader/assembler pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.llm.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class ChunkFallbackStrategy:
    """Falls back to existing pgvector chunk search."""

    def __init__(
        self,
        vector_db: Any,
        embedder: "EmbeddingProvider",
        config: "FitzKragConfig",
    ):
        self._vector_db = vector_db
        self._embedder = embedder
        self._config = config

    def retrieve(self, query: str, limit: int) -> list[Address]:
        """Search chunks via vector DB and wrap as CHUNK addresses."""
        try:
            query_vector = self._embedder.embed(query)
            results = self._vector_db.search(
                collection_name=self._config.collection,
                query_vector=query_vector,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"Chunk fallback search failed: {e}")
            return []

        return [self._to_chunk_address(r) for r in results]

    def _to_chunk_address(self, r: Any) -> Address:
        """Convert a vector DB result to a CHUNK address."""
        metadata = getattr(r, "metadata", {}) if hasattr(r, "metadata") else {}
        return Address(
            kind=AddressKind.CHUNK,
            source_id=getattr(r, "id", str(r)),
            location=metadata.get("source_file", "unknown"),
            summary=metadata.get("summary", ""),
            score=getattr(r, "score", 0.0),
            metadata={
                "chunk_id": getattr(r, "id", str(r)),
                "text": getattr(r, "text", ""),
                **metadata,
            },
        )
