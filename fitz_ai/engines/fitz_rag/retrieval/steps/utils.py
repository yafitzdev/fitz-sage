# fitz_ai/engines/fitz_rag/retrieval/steps/utils.py
"""Utility functions for retrieval steps."""

from __future__ import annotations

from typing import Any

from fitz_ai.core.chunk import Chunk


def normalize_record(record: Any, extra_metadata: dict[str, Any] | None = None) -> Chunk:
    """
    Convert a vector DB record to a Chunk object.

    Handles both dict-style and object-style records from various vector DB clients.
    Flattens nested metadata structure (where chunk.metadata is stored under "metadata" key).

    Args:
        record: Vector DB record (dict or object with attributes)
        extra_metadata: Additional metadata to merge into the chunk

    Returns:
        Normalized Chunk object
    """
    # Extract record ID
    record_id = record.get("id") if isinstance(record, dict) else getattr(record, "id", None)

    # Extract payload (may be nested under "payload" or "metadata")
    payload = (
        record.get("payload", {}) if isinstance(record, dict) else getattr(record, "payload", {})
    )

    # Flatten nested metadata
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        flat_metadata = {**payload, **metadata}
    else:
        flat_metadata = payload

    # Merge extra metadata if provided
    if extra_metadata:
        flat_metadata = {**flat_metadata, **extra_metadata}

    # Build Chunk
    chunk = Chunk(
        id=str(record_id),
        doc_id=str(payload.get("doc_id", "unknown")),
        content=str(payload.get("content", "")),
        chunk_index=int(payload.get("chunk_index", 0)),
        metadata=flat_metadata,
    )

    return chunk
