# fitz_ai/engines/fitz_rag/protocols.py
"""
Protocol definitions for duck-typed interfaces.

These are documentation-only type hints. DO NOT check isinstance() against them.
Just call the methods and let duck typing work.
"""

from __future__ import annotations

from typing import Any, Protocol


class VectorClient(Protocol):
    """Vector database interface. Don't check isinstance - just call the methods."""

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
        query_filter: dict | None = None,
    ) -> list[Any]:
        """Search for similar vectors."""
        ...

    def retrieve(
        self,
        collection: str,
        ids: list[str],
        with_payload: bool = True,
    ) -> list[Any]:
        """Retrieve specific records by ID."""
        ...


class Embedder(Protocol):
    """Embedding interface. Don't check isinstance - just call the methods."""

    def embed(self, query: str) -> list[float]:
        """Embed text into a vector."""
        ...


class ChatClient(Protocol):
    """Chat LLM interface. Don't check isinstance - just call the methods."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Send chat messages and get a response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Generated text response
        """
        ...
