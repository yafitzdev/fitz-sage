# fitz_ai/ingest/chunking/plugins/overlap.py
"""
Overlap chunker - character-based chunking with overlap.

This is essentially an alias for SimpleChunker with a different plugin_name,
provided for backwards compatibility and clarity when overlap is the primary
concern.

Chunker ID format: "overlap:{chunk_size}:{chunk_overlap}"
Example: "overlap:1000:200" for 1000-char chunks with 200-char overlap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from fitz_ai.engines.classic_rag.models.chunk import Chunk


@dataclass
class OverlapChunker:
    """
    Character-based chunker with configurable overlap.

    Identical to SimpleChunker but with a distinct plugin_name for
    cases where you want to explicitly use overlap chunking.

    The overlap helps maintain context across chunk boundaries, which
    can improve retrieval quality for queries that span chunk boundaries.

    Example:
        >>> chunker = OverlapChunker(chunk_size=1000, chunk_overlap=200)
        >>> chunker.chunker_id
        'overlap:1000:200'
    """

    plugin_name: str = field(default="overlap", repr=False)
    chunk_size: int = 1000
    chunk_overlap: int = 200  # Default to 200 for overlap chunker

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )

    @property
    def chunker_id(self) -> str:
        """
        Unique identifier for this chunker configuration.

        Format: "overlap:{chunk_size}:{chunk_overlap}"
        """
        return f"{self.plugin_name}:{self.chunk_size}:{self.chunk_overlap}"

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        """
        Split text into overlapping character chunks.

        Args:
            text: The text content to chunk.
            base_meta: Base metadata to include in each chunk.

        Returns:
            List of Chunk objects with overlapping content.
        """
        if not text or not text.strip():
            return []

        doc_id = str(
            base_meta.get("doc_id") or base_meta.get("source_file") or "unknown"
        )

        chunks: List[Chunk] = []
        chunk_index = 0

        # Step size: how far to advance after each chunk
        step = self.chunk_size - self.chunk_overlap
        if step < 1:
            step = 1

        pos = 0
        while pos < len(text):
            end = pos + self.chunk_size
            piece = text[pos:end].strip()

            if piece:
                chunk_id = f"{doc_id}:{chunk_index}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        content=piece,
                        metadata=dict(base_meta),
                    )
                )
                chunk_index += 1

            pos += step

        return chunks


__all__ = ["OverlapChunker"]