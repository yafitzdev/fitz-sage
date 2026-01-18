# fitz_ai/engines/fitz_rag/retrieval/multihop/utils.py
"""Utility functions for multi-hop retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk


def build_context_from_chunks(
    chunks: list["Chunk"],
    max_chars: int = 5000,
    chunk_truncate_chars: int = 500,
) -> str:
    """
    Build a context string from chunks, respecting max length.

    Args:
        chunks: List of chunks to build context from
        max_chars: Maximum total characters for the context
        chunk_truncate_chars: Maximum characters per individual chunk

    Returns:
        Context string with chunks separated by double newlines
    """
    context_parts: list[str] = []
    total_chars = 0

    for chunk in chunks:
        content = chunk.content[:chunk_truncate_chars]
        if total_chars + len(content) > max_chars:
            break
        context_parts.append(content)
        total_chars += len(content) + 2  # +2 for newlines

    return "\n\n".join(context_parts)
