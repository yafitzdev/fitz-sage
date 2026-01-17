# fitz_ai/core/chunk.py
"""Chunk - fundamental unit of knowledge. See docs/api_reference.md for examples."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class ChunkLike(Protocol):
    """Protocol for duck-typed chunk handling."""

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
    """Canonical chunk model: document segment with ID, content, and metadata."""

    id: str = Field(..., description="Chunk ID")
    doc_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Index of this chunk within its document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


__all__ = ["Chunk", "ChunkLike"]
