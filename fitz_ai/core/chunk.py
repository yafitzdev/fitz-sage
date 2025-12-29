# fitz_ai/core/chunk.py
"""
Chunk - Core data model for fitz-ai.

The Chunk is the fundamental unit of knowledge in fitz-ai. All engines,
ingestion pipelines, and vector stores work with chunks.

This module provides:
- Chunk: The canonical Pydantic model
- ChunkLike: Protocol for duck-typed chunk handling
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class ChunkLike(Protocol):
    """
    Protocol for chunk-like objects.

    Use this when you need duck-typed chunk handling without
    requiring the concrete Chunk class.
    """

    @property
    def id(self) -> str: ...

    @property
    def doc_id(self) -> str: ...

    @property
    def chunk_index(self) -> int: ...

    @property
    def content(self) -> str: ...

    @property
    def metadata(self) -> Dict[str, Any] | None: ...


class Chunk(BaseModel):
    """
    Canonical chunk model used across the entire fitz stack.

    A chunk is a segment of a document with:
    - Unique identifier
    - Reference to parent document
    - Position within the document
    - Text content
    - Optional metadata
    """

    id: str = Field(..., description="Chunk ID")
    doc_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Index of this chunk within its document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


__all__ = ["Chunk", "ChunkLike"]
