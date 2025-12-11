from __future__ import annotations
from typing import List, Dict

from fitz_ingest.chunker.base import BaseChunker, Chunk


class SimpleChunker(BaseChunker):
    """
    Basic fixed-size character chunker:

    - Splits text into N-character blocks
    - No overlap
    - No logging (delegated to ChunkingEngine)
    """

    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def chunk_text(self, text: str, base_meta: Dict) -> List[Chunk]:
        chunks = []
        size = self.chunk_size
        length = len(text)

        for i in range(0, length, size):
            piece = text[i:i + size].strip()
            if not piece:
                continue

            chunks.append(
                Chunk(
                    text=piece,
                    metadata=dict(base_meta),
                )
            )

        return chunks
