"""
Simple fixed-size chunkers for fitz-rag.
Splits text into chunks of 500 characters with no overlap.
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
    Very simple chunkers that:
    - Reads a file as text
    - Splits it into 500-character chunks
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
    # Chunk pure text
    # ---------------------------------------------------------
    def _chunk_text(self, text: str, base_meta: dict) -> List[Chunk]:
        chunks = []
        size = self.chunk_size
        length = len(text)

        for i in range(0, length, size):
            chunk_text = text[i:i + size].strip()
            if not chunk_text:
                continue

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=dict(base_meta),
                )
            )

        return chunks
