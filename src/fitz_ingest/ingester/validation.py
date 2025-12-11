from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class IngestionValidationError(Exception):
    """Raised when ingested chunks fail validation."""


@dataclass
class ChunkValidationConfig:
    """
    Configuration for validating ingested chunks.
    """
    min_chars: int = 1
    max_chars: int = 20000
    required_metadata_keys: tuple[str, ...] = ()


class IngestionValidator:
    """
    Validates ingested chunks before they are sent to the vector DB.
    Supports both dict-based and legacy object-based chunks via normalization.
    """

    def __init__(self, cfg: ChunkValidationConfig | None = None) -> None:
        self.cfg = cfg or ChunkValidationConfig()

    # -------------------------------
    # Public API
    # -------------------------------
    def validate_chunks(self, chunks: Iterable[Any], file_path: str) -> None:
        """
        Validate a sequence of chunks for a given file.

        Raises IngestionValidationError on first error.
        """
        file_str = str(file_path)

        for idx, chunk in enumerate(chunks):
            self._validate_single_chunk(chunk, idx, file_str)

    # -------------------------------
    # Internal helpers
    # -------------------------------
    def _as_dict_chunk(self, chunk: Any) -> dict:
        """
        Normalize a chunk-like object into a dict with at least:
        - 'text'
        - 'metadata'

        Accepts:
        - dict with text/metadata
        - legacy objects with .text and .metadata
        """
        if isinstance(chunk, dict):
            return chunk

        text = getattr(chunk, "text", None)
        metadata = getattr(chunk, "metadata", None)

        return {
            "text": text,
            "metadata": metadata,
        }

    def _validate_single_chunk(self, chunk: Any, idx: int, file_str: str) -> None:
        c = self._as_dict_chunk(chunk)

        # ---- text ----
        text = c.get("text", None)
        if not isinstance(text, str):
            raise IngestionValidationError(
                f"Chunk {idx} in '{file_str}' has non-string text (type={type(text)})"
            )

        # ---- length ----
        length = len(text)
        if length < self.cfg.min_chars or length > self.cfg.max_chars:
            raise IngestionValidationError(
                f"Chunk {idx} in '{file_str}' has invalid text length (len={length})"
            )

        # ---- metadata type ----
        metadata = c.get("metadata", None)
        if not isinstance(metadata, Mapping):
            raise IngestionValidationError(
                f"Chunk {idx} in '{file_str}' has invalid metadata type (type={type(metadata)})"
            )

        # ---- required metadata keys ----
        for key in self.cfg.required_metadata_keys:
            if key not in metadata:
                raise IngestionValidationError(
                    f"Chunk {idx} in '{file_str}' is missing required metadata key '{key}'"
                )
