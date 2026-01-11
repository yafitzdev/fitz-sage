# fitz_ai/ingestion/chunking/plugins/default/simple.py
"""
Simple fixed-size character chunker.

Splits text into fixed-size character blocks with optional overlap.
This is the default/fallback chunker used when no file-type specific
chunker is configured.

Chunker ID format: "simple:{chunk_size}:{chunk_overlap}"
Example: "simple:1000:0" for 1000-char chunks with no overlap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from fitz_ai.core.chunk import Chunk
from fitz_ai.core.document import ParsedDocument


@dataclass
class SimpleChunker:
    """
    Fixed-size character chunker with optional overlap.

    Example:
        >>> chunker = SimpleChunker(chunk_size=500, chunk_overlap=50)
        >>> chunker.chunker_id
        'simple:500:50'
        >>> chunks = chunker.chunk(parsed_document)
    """

    plugin_name: str = field(default="simple", repr=False)
    chunk_size: int = 1000
    chunk_overlap: int = 0

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

        Format: "simple:{chunk_size}:{chunk_overlap}"
        """
        return f"{self.plugin_name}:{self.chunk_size}:{self.chunk_overlap}"

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        """
        Split a parsed document into fixed-size character chunks.

        Args:
            document: ParsedDocument with structured elements.

        Returns:
            List of Chunk objects. Empty list if document is empty.
        """
        # Use full text for simple character-based chunking
        text = document.full_text
        if not text or not text.strip():
            return []

        # Extract doc_id from document
        doc_id = document.metadata.get("doc_id")
        if not doc_id:
            # Derive from source path
            source_path = Path(document.source.replace("file:///", ""))
            doc_id = source_path.stem if source_path.stem else "unknown"

        # Build base metadata
        base_meta = {
            "source_file": document.source,
            "doc_id": doc_id,
            **document.metadata,
        }

        chunks: List[Chunk] = []
        chunk_index = 0

        step = self.chunk_size - self.chunk_overlap
        if step < 1:
            step = 1

        pos = 0
        while pos < len(text):
            end = pos + self.chunk_size
            piece = text[pos:end].strip()

            # Skip tiny trailing chunks that are already covered by previous overlap
            # This prevents creating near-duplicate chunks at document end
            if piece and (chunk_index == 0 or len(piece) > self.chunk_overlap):
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


__all__ = ["SimpleChunker"]
