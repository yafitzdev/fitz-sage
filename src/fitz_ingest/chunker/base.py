"""
Simple fixed-size chunkers for fitz-rag.
Splits text into chunks of 500 characters with no overlap.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import CHUNKING

logger = get_logger(__name__)


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

        logger.debug(f"{CHUNKING} Chunking file: {file_path}")

        if not path.exists() or not path.is_file():
            logger.error(f"{CHUNKING} File does not exist or is not a file: {file_path}")
            return []

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"{CHUNKING} Failed reading file '{file_path}': {e}")
            return []

        chunks = self._chunk_text(text, {"source_file": str(path)})

        logger.debug(f"{CHUNKING} Extracted {len(chunks)} chunks from '{file_path}'")
        return chunks

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
