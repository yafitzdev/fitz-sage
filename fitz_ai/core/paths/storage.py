# fitz_ai/core/paths/storage.py
"""Engine storage paths (vector DB, GraphRAG, CLaRA)."""

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


def graphrag_storage(collection: str) -> Path:
    """
    GraphRAG knowledge graph storage path.

    Location: {workspace}/graphrag/{collection}.json
    """
    return workspace() / "graphrag" / f"{collection}.json"


def ensure_graphrag_storage() -> Path:
    """Get graphrag directory and create it if it doesn't exist."""
    path = workspace() / "graphrag"
    path.mkdir(parents=True, exist_ok=True)
    return path


def clara_storage(collection: str) -> Path:
    """
    CLaRA compressed representations storage path.

    Location: {workspace}/clara/{collection}/
    """
    return workspace() / "clara" / collection


def ensure_clara_storage(collection: str) -> Path:
    """Get clara collection directory and create it if it doesn't exist."""
    path = clara_storage(collection)
    path.mkdir(parents=True, exist_ok=True)
    return path
