"""
Validation utilities for fitz-ingest.

Used by the ingestion engine to ensure that, before writing to Qdrant:

- Chunks are structurally valid
- Chunk sizes are within reasonable bounds
- Metadata has required keys (if configured)

This is intentionally light-weight for v0.1.0 and can be extended later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Any, Sequence, Optional

from fitz_ingest.exceptions.chunking import IngestionChunkingError
from fitz_ingest.exceptions.config import IngestionConfigError


class IngestionValidationError(IngestionChunkingError):
    pass


@dataclass
class ChunkValidationConfig:
    min_chars: int = 1
    max_chars: int = 50_000
    required_metadata_keys: Sequence[str] = ()


class IngestionValidator:
    def __init__(self, config: Optional[ChunkValidationConfig] = None) -> None:
        self.config = config or ChunkValidationConfig()

    def validate_chunks(self, chunks: Iterable[Any], file_path: str | Path) -> None:
        file_str = str(file_path)
        for idx, chunk in enumerate(chunks):
            self._validate_single_chunk(chunk, idx, file_str)

    def _validate_single_chunk(self, chunk: Any, idx: int, file_str: str) -> None:
        # Text
        text = getattr(chunk, "text", None)
        if not isinstance(text, str):
            raise IngestionValidationError(
                f"Chunk {idx} in '{file_str}' has non-string text (type={type(text)})"
            )

        length = len(text)
        if length < self.config.min_chars or length > self.config.max_chars:
            raise IngestionValidationError(
                f"Chunk {idx} in '{file_str}' has invalid text length {length} "
                f"(allowed range: {self.config.min_chars}-{self.config.max_chars})"
            )

        # Metadata
        metadata = getattr(chunk, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise IngestionValidationError(
                f"Chunk {idx} in '{file_str}' has invalid metadata type "
                f"(expected Mapping, got {type(metadata)})"
            )

        # Required keys
        for key in self.config.required_metadata_keys:
            if key not in metadata:
                raise IngestionValidationError(
                    f"Chunk {idx} in '{file_str}' is missing required metadata key '{key}'"
                )
