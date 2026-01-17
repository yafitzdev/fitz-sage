# fitz_ai/core/chunk.py
"""Chunk - fundamental unit of knowledge. See docs/api_reference.md for examples."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Canonical chunk model: document segment with ID, content, and metadata."""

    id: str = Field(..., description="Chunk ID")
    doc_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Index of this chunk within its document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


__all__ = ["Chunk"]
