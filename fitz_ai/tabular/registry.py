# fitz_ai/tabular/registry.py
"""
Table chunk ID registry.

Stores table schema chunk IDs at ingestion time so they can be
fetched directly at query time without scrolling or searching.
"""

from __future__ import annotations

import json
from pathlib import Path

from fitz_ai.core.paths import FitzPaths


def get_table_ids(collection: str) -> list[str]:
    """Get all table chunk IDs for a collection."""
    path = FitzPaths.table_registry(collection)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data.get("table_ids", [])
    except (json.JSONDecodeError, OSError):
        return []


def add_table_id(collection: str, table_id: str) -> None:
    """Add a table chunk ID to the registry."""
    FitzPaths.ensure_table_registry_dir()
    path = FitzPaths.table_registry(collection)

    ids = get_table_ids(collection)
    if table_id not in ids:
        ids.append(table_id)
        path.write_text(json.dumps({"table_ids": ids}, indent=2))


def remove_table_id(collection: str, table_id: str) -> None:
    """Remove a table chunk ID from the registry."""
    path = FitzPaths.table_registry(collection)
    if not path.exists():
        return

    ids = get_table_ids(collection)
    if table_id in ids:
        ids.remove(table_id)
        path.write_text(json.dumps({"table_ids": ids}, indent=2))


def clear_tables(collection: str) -> None:
    """Clear all table IDs for a collection."""
    path = FitzPaths.table_registry(collection)
    if path.exists():
        path.unlink()


__all__ = ["get_table_ids", "add_table_id", "remove_table_id", "clear_tables"]
