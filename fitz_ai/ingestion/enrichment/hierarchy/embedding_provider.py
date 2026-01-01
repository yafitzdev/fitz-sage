# fitz_ai/ingestion/enrichment/hierarchy/embedding_provider.py
"""
Embedding provider for semantic grouping.

Computes embeddings for chunks using the configured embedder.
Used by SemanticGrouper during hierarchy enrichment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding clients."""

    def embed(self, text: str) -> list[float]: ...


class EmbeddingProvider:
    """
    Provides embeddings for chunks during semantic grouping.

    Computes embeddings on-demand using the configured embedder.
    """

    def __init__(self, embedder: Embedder):
        """
        Initialize embedding provider.

        Args:
            embedder: Embedding client implementing embed(text) -> list[float].
        """
        self._embedder = embedder

    def get_embeddings(self, chunks: List["Chunk"]) -> np.ndarray:
        """
        Compute embeddings for a list of chunks.

        Args:
            chunks: List of chunks to embed.

        Returns:
            (N, D) numpy array of embeddings aligned with chunks.
        """
        if not chunks:
            return np.array([]).reshape(0, 0)

        logger.info(f"[SEMANTIC] Computing embeddings for {len(chunks)} chunks")

        embeddings = []
        for chunk in chunks:
            vector = self._embedder.embed(chunk.content)
            embeddings.append(vector)

        return np.array(embeddings, dtype=np.float32)


__all__ = ["EmbeddingProvider", "Embedder"]
