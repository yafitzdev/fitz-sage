"""
Simple fixed-size chunkers for fitz_ingest.

This chunkers:
- Splits text into 500-character chunks
- No overlap
- Metadata includes source file
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    text: str
    metadata: dict


class SimpleChunker:
    """
    Minimal file chunkers for ingestion.
    """

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size

    # ---------------------------------------------------------
    # Chunk a file
    # ---------------------------------------------------------
    def chunk_file(self, file_path: str) -> List[Chunk]:
        path = Path(file_path)

        if not path.exists() or not path.is_file():
            return []

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        return self._chunk_text(text, {"source_file": str(path)})

    # ---------------------------------------------------------
    # Internal text chunking
    # ---------------------------------------------------------
    def _chunk_text(self, text: str, base_meta: dict) -> List[Chunk]:
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
