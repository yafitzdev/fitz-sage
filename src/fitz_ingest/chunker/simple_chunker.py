from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

from fitz_rag.exceptions.config import ConfigError


@dataclass
class Chunk:
    text: str
    metadata: dict


class SimpleChunker:
    """
    Very simple chunker:
    - Reads file as UTF-8 text
    - Splits into fixed-size segments
    """

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size

    # ---------------------------------------------------------
    # Chunk a file
    # ---------------------------------------------------------
    def chunk_file(self, file_path: str) -> List[Chunk]:
        path = Path(file_path)

        # Invalid path → empty list (your original design)
        if not path.exists() or not path.is_file():
            return []

        # Try reading text
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            # Return [] as originally, but also raise structured error
            raise ConfigError(f"Failed reading file for chunking: {file_path}") from e

        # Delegate chunking
        try:
            return self._chunk_text(text, {"source_file": str(path)})
        except Exception as e:
            raise ConfigError(f"Failed chunking text from file: {file_path}") from e

    # ---------------------------------------------------------
    # Internal text chunking
    # ---------------------------------------------------------
    def _chunk_text(self, text: str, base_meta: dict) -> List[Chunk]:
        try:
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

        except Exception as e:
            raise ConfigError("Unexpected failure inside chunking logic") from e
