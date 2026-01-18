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


def parse_json_list(response: str, max_items: int | None = None) -> list[str]:
    """
    Parse a JSON array of strings from an LLM response.

    Handles common LLM response formats:
    - Plain JSON array
    - Markdown code blocks with or without "json" language tag
    - Falls back to newline-split if JSON parsing fails

    Args:
        response: Raw LLM response text
        max_items: Maximum items to return (None for unlimited)

    Returns:
        List of strings extracted from the response
    """
    import json

    text = response.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()

    # Try JSON parsing
    try:
        result = json.loads(text)
        if isinstance(result, list):
            items = [str(item) for item in result]
            return items[:max_items] if max_items else items
    except json.JSONDecodeError:
        pass

    # Fallback: split by newlines
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    return lines[:max_items] if max_items else lines
