# fitz_ai/core/paths/storage.py
"""Engine storage paths (vector DB)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .workspace import workspace


def vector_db(collection: Optional[str] = None) -> Path:
    """
    Local vector database storage path.

    Location: {workspace}/vector_db/
    Or with collection: {workspace}/vector_db/{collection}/
    """
    base = workspace() / "vector_db"
    if collection:
        return base / collection
    return base


def ensure_vector_db(collection: Optional[str] = None) -> Path:
    """Get vector DB path and create it if it doesn't exist."""
    path = vector_db(collection)
    path.mkdir(parents=True, exist_ok=True)
    return path


def pgdata(collection: Optional[str] = None) -> Path:
    """
    PostgreSQL data directory for pgserver.

    Location: {workspace}/pgdata/
    Or with collection: {workspace}/pgdata/{collection}/
    """
    base = workspace() / "pgdata"
    if collection:
        return base / collection
    return base


def ensure_pgdata(collection: Optional[str] = None) -> Path:
    """Get pgdata path and create it if it doesn't exist."""
    path = pgdata(collection)
    path.mkdir(parents=True, exist_ok=True)
    return path
