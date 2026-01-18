# fitz_ai/core/paths/indices.py
"""Retrieval index paths (vocabulary, sparse index, entity graph)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .workspace import workspace


def vocabulary(collection: Optional[str] = None) -> Path:
    """
    Auto-detected keyword vocabulary file.

    Location: {workspace}/keywords.yaml (default/global)
    Or with collection: {workspace}/keywords/{collection}.yaml
    """
    if collection:
        return workspace() / "keywords" / f"{collection}.yaml"
    return workspace() / "keywords.yaml"


def sparse_index(collection: str) -> Path:
    """
    Sparse (TF-IDF) index for hybrid search.

    Location: {workspace}/sparse_index/{collection}
    """
    return workspace() / "sparse_index" / collection


def ensure_sparse_index_dir() -> Path:
    """Get sparse index directory and create it if it doesn't exist."""
    path = workspace() / "sparse_index"
    path.mkdir(parents=True, exist_ok=True)
    return path


def entity_graph(collection: str) -> Path:
    """
    Entity graph database for a collection.

    Location: {workspace}/entity_graph/{collection}.db
    """
    return workspace() / "entity_graph" / f"{collection}.db"


def ensure_entity_graph_dir() -> Path:
    """Get entity graph directory and create it if it doesn't exist."""
    path = workspace() / "entity_graph"
    path.mkdir(parents=True, exist_ok=True)
    return path
