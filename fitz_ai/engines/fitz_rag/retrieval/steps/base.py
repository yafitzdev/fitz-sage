# fitz_ai/engines/fitz_rag/retrieval/steps/base.py
"""
Base classes and protocols for retrieval steps.

All retrieval steps inherit from RetrievalStep and implement execute().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from fitz_ai.core.chunk import Chunk

# =============================================================================
# Protocols for Dependencies
# =============================================================================


@runtime_checkable
class VectorClient(Protocol):
    """Protocol for vector database clients."""

    def search(self, *args: Any, **kwargs: Any) -> list[Any]: ...

    def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding services."""

    def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking services."""

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]: ...


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for chat services (used for query expansion)."""

    def chat(self, messages: list[dict[str, Any]]) -> str: ...


@runtime_checkable
class KeywordMatcherClient(Protocol):
    """Protocol for keyword matching services (used for exact keyword filtering)."""

    def find_in_query(self, query: str) -> list[Any]: ...

    def chunk_matches_any(self, chunk: Any, keywords: list[Any] | None = None) -> bool: ...


@runtime_checkable
class EntityGraphClient(Protocol):
    """Protocol for entity graph services (used for related chunk discovery)."""

    def get_related_chunks(
        self,
        chunk_ids: list[str],
        max_total: int = 20,
        min_shared_entities: int = 1,
    ) -> list[str]: ...


# =============================================================================
# Base Step
# =============================================================================


@dataclass
class RetrievalStep(ABC):
    """
    Base class for retrieval steps.

    All steps take a query and list of chunks, and return an updated list of chunks.
    Steps are stateless and composable.
    """

    @abstractmethod
    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Execute step and return updated chunks."""
        ...

    @property
    def name(self) -> str:
        """Return the step class name."""
        return self.__class__.__name__
