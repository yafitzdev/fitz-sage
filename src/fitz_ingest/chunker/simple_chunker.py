from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

from fitz_rag.exceptions.config import ConfigError

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import CHUNKING

logger = get_logger(__name__)


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

        logger.debug(f"{CHUNKING} Chunking file: {file_path}")

        # Invalid path → empty list (your original design)
        if not path.exists() or not path.is_file():
            logger.error(f"{CHUNKING} File does not exist or is not a file: {file_path}")
            return []

        # Try reading text
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"{CHUNKING} Failed reading file '{file_path}': {e}")
            raise ConfigError(f"Failed reading file for chunking: {file_path}") from e

        # Delegate chunking
        try:
            chunks = self._chunk_text(text, {"source_file": str(path)})
            logger.debug(f"{CHUNKING} Extracted {len(chunks)} chunks from '{file_path}'")
            return chunks
        except Exception as e:
            logger.error(f"{CHUNKING} Failed chunking text from file '{file_path}': {e}")
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
            logger.error(f"{CHUNKING} Unexpected failure in chunking logic: {e}")
            raise ConfigError("Unexpected failure inside chunking logic") from e
