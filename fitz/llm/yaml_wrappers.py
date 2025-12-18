# fitz/llm/yaml_wrappers.py
"""
Wrapper classes for YAML plugins.

These wrappers allow YAML plugins to be used via the registry.
"""
from __future__ import annotations

from typing import Any, Type


def create_yaml_plugin_wrapper(plugin_type: str, plugin_name: str) -> Type[Any]:
    """
    Create a wrapper class for a YAML plugin.

    Args:
        plugin_type: Type of plugin ("chat", "embedding", "rerank")
        plugin_name: Name of the YAML plugin

    Returns:
        A class that wraps the appropriate YAML client
    """
    if plugin_type == "chat":
        return _create_chat_wrapper(plugin_name)
    elif plugin_type == "embedding":
        return _create_embedding_wrapper(plugin_name)
    elif plugin_type == "rerank":
        return _create_rerank_wrapper(plugin_name)
    else:
        raise ValueError(f"Invalid plugin type: {plugin_type!r}")


def _create_chat_wrapper(name: str) -> Type[Any]:
    """Create wrapper for chat plugin."""
    from fitz.llm.runtime import YAMLChatClient

    _plugin_name = name

    class ChatWrapper:
        plugin_name = _plugin_name
        plugin_type = "chat"

        def __init__(self, **kwargs: Any) -> None:
            self._client = YAMLChatClient.from_name(_plugin_name, **kwargs)

        def chat(self, messages: list[dict[str, Any]]) -> str:
            return self._client.chat(messages)

    ChatWrapper.__name__ = f"Chat_{name}"
    ChatWrapper.__qualname__ = f"Chat_{name}"
    return ChatWrapper


def _create_embedding_wrapper(name: str) -> Type[Any]:
    """Create wrapper for embedding plugin."""
    from fitz.llm.runtime import YAMLEmbeddingClient

    _plugin_name = name

    class EmbeddingWrapper:
        plugin_name = _plugin_name
        plugin_type = "embedding"

        def __init__(self, **kwargs: Any) -> None:
            self._client = YAMLEmbeddingClient.from_name(_plugin_name, **kwargs)

        def embed(self, text: str) -> list[float]:
            return self._client.embed(text)

    EmbeddingWrapper.__name__ = f"Embedding_{name}"
    EmbeddingWrapper.__qualname__ = f"Embedding_{name}"
    return EmbeddingWrapper


def _create_rerank_wrapper(name: str) -> Type[Any]:
    """Create wrapper for rerank plugin.

    The rerank wrapper handles conversion between Chunk objects and strings.
    The YAML client expects list[str] and returns list[tuple[int, float]].
    The retriever passes list[Chunk] and expects list[Chunk] back.
    """
    from fitz.llm.runtime import YAMLRerankClient

    _plugin_name = name

    class RerankWrapper:
        plugin_name = _plugin_name
        plugin_type = "rerank"

        def __init__(self, **kwargs: Any) -> None:
            self._client = YAMLRerankClient.from_name(_plugin_name, **kwargs)

        def rerank(
                self,
                query: str,
                documents: list[Any],
                top_n: int | None = None,
        ) -> list[Any]:
            """Rerank documents/chunks by relevance to query.

            Handles both:
            - list[str] -> returns list[tuple[int, float]]
            - list[Chunk] -> returns list[Chunk] (reordered)
            """
            if not documents:
                return []

            # Check if we have Chunk objects or strings
            first = documents[0]
            is_chunk = hasattr(first, 'content')

            if is_chunk:
                # Extract text from chunks
                texts = [getattr(doc, 'content', str(doc)) for doc in documents]

                # Call the YAML client
                ranked = self._client.rerank(query, texts, top_n)

                # Reorder chunks by ranked indices
                return [documents[idx] for idx, _score in ranked]
            else:
                # Plain strings - return tuples directly
                return self._client.rerank(query, documents, top_n)

    RerankWrapper.__name__ = f"Rerank_{name}"
    RerankWrapper.__qualname__ = f"Rerank_{name}"
    return RerankWrapper