# fitz_ai/cli/commands/ingest_adapters.py
"""
Adapter classes for ingest command.

Protocol adapters for bridging different interfaces.
"""

from __future__ import annotations

from typing import Any, Dict, List


class VectorDBWriterAdapter:
    """Adapts vector DB client to VectorDBWriter protocol."""

    def __init__(self, client):
        self._client = client

    def upsert(
        self, collection: str, points: List[Dict[str, Any]], defer_persist: bool = False
    ) -> None:
        """Upsert points into collection."""
        self._client.upsert(collection, points, defer_persist=defer_persist)

    def flush(self) -> None:
        """Flush any pending writes to disk."""
        if hasattr(self._client, "flush"):
            self._client.flush()
