# fitz_ai/llm/client.py
"""
Public API for LLM providers.

Simple entry points for getting chat, embedding, rerank, and vision providers.
"""

from __future__ import annotations

from typing import Any

from fitz_ai.llm.config import (
    create_chat_provider,
    create_embedding_provider,
    create_rerank_provider,
    create_vision_provider,
)
from fitz_ai.llm.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    ModelTier,
    RerankProvider,
    VisionProvider,
)


def get_chat(
    spec: str,
    tier: ModelTier = "smart",
    config: dict[str, Any] | None = None,
) -> ChatProvider:
    """
    Get a chat provider.

    Args:
        spec: Provider spec like "cohere" or "cohere/command-a-03-2025"
        tier: Model tier (smart, balanced, fast)
        config: Optional config with auth/base_url settings

    Returns:
        ChatProvider instance

    Examples:
        >>> chat = get_chat("cohere")
        >>> chat = get_chat("cohere", tier="fast")
        >>> chat = get_chat("openai/gpt-4o")
        >>> response = chat.chat([{"role": "user", "content": "Hello"}])
    """
    return create_chat_provider(spec, config, tier)


def get_embedder(
    spec: str,
    config: dict[str, Any] | None = None,
) -> EmbeddingProvider:
    """
    Get an embedding provider.

    Args:
        spec: Provider spec like "cohere" or "cohere/embed-multilingual-v3.0"
        config: Optional config with auth/dimensions settings

    Returns:
        EmbeddingProvider instance

    Examples:
        >>> embedder = get_embedder("cohere")
        >>> vector = embedder.embed("Hello world")
        >>> vectors = embedder.embed_batch(["Hello", "World"])
    """
    return create_embedding_provider(spec, config)


def get_reranker(
    spec: str | None,
    config: dict[str, Any] | None = None,
) -> RerankProvider | None:
    """
    Get a rerank provider.

    Args:
        spec: Provider spec like "cohere" or None to disable
        config: Optional config with auth settings

    Returns:
        RerankProvider instance, or None if spec is None

    Examples:
        >>> reranker = get_reranker("cohere")
        >>> if reranker:
        ...     results = reranker.rerank("query", ["doc1", "doc2"])
    """
    return create_rerank_provider(spec, config)


def get_vision(
    spec: str | None,
    config: dict[str, Any] | None = None,
) -> VisionProvider | None:
    """
    Get a vision provider.

    Args:
        spec: Provider spec like "openai/gpt-4o" or None to disable
        config: Optional config with auth settings

    Returns:
        VisionProvider instance, or None if spec is None

    Examples:
        >>> vision = get_vision("openai/gpt-4o")
        >>> if vision:
        ...     description = vision.describe_image(base64_data)
    """
    return create_vision_provider(spec, config)


__all__ = [
    "get_chat",
    "get_embedder",
    "get_reranker",
    "get_vision",
]
