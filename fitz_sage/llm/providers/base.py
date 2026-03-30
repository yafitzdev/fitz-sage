# fitz_sage/llm/providers/base.py
"""
Provider protocols for LLM clients.

All providers implement these protocols for type-safe usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol, runtime_checkable

ModelTier = Literal["fast", "balanced", "smart"]


@dataclass
class RerankResult:
    """Result from reranking a document."""

    index: int
    score: float


@runtime_checkable
class ChatProvider(Protocol):
    """Protocol for chat/completion providers."""

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """
        Generate a chat completion.

        Args:
            messages: List of messages with 'role' and 'content' keys.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)

        Returns:
            Generated text response.
        """
        ...


@runtime_checkable
class StreamingChatProvider(Protocol):
    """Protocol for chat providers that support streaming."""

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """
        Generate a streaming chat completion.

        Args:
            messages: List of messages with 'role' and 'content' keys.
            **kwargs: Provider-specific options.

        Yields:
            Text chunks as they are generated.
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str, *, task_type: str | None = None) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed.
            task_type: "query" for retrieval queries, "document" for ingestion.
                       None uses provider default.

        Returns:
            Embedding vector.
        """
        ...

    def embed_batch(self, texts: list[str], *, task_type: str | None = None) -> list[list[float]]:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed.
            task_type: "query" for retrieval queries, "document" for ingestion.
                       None uses provider default.

        Returns:
            List of embedding vectors (same order as input).
        """
        ...

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        ...


@runtime_checkable
class RerankProvider(Protocol):
    """Protocol for reranking providers."""

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Query to rank against.
            documents: List of document texts.
            top_n: Maximum number of results to return.

        Returns:
            List of RerankResult sorted by relevance (highest first).
        """
        ...


@runtime_checkable
class VisionProvider(Protocol):
    """Protocol for vision/image description providers."""

    def describe_image(self, image_base64: str, prompt: str | None = None) -> str:
        """
        Describe an image using a vision model.

        Args:
            image_base64: Base64-encoded image data.
            prompt: Custom prompt for description.

        Returns:
            Text description of the image.
        """
        ...


__all__ = [
    "ModelTier",
    "RerankResult",
    "ChatProvider",
    "StreamingChatProvider",
    "EmbeddingProvider",
    "RerankProvider",
    "VisionProvider",
]
