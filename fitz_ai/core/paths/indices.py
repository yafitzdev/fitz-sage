# fitz_ai/core/paths/indices.py
"""
Retrieval index paths.

DEPRECATED: All indices are now stored in PostgreSQL.
- vocabulary → keywords table in collection database
- sparse_index → tsvector column in chunks table (auto-maintained)
- entity_graph → entities + entity_chunks tables in collection database

These functions are kept for backwards compatibility during migration
but will be removed in a future version.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from .workspace import workspace


def vocabulary(collection: Optional[str] = None) -> Path:
    """
    DEPRECATED: Vocabulary is now stored in PostgreSQL.

    This function returns the old YAML file path for backwards compatibility
    during migration. It will be removed in a future version.
    """
    warnings.warn(
        "vocabulary() path is deprecated. Vocabulary is now stored in PostgreSQL.",
        DeprecationWarning,
        stacklevel=2,
    )
    if collection:
        return workspace() / "keywords" / f"{collection}.yaml"
    return workspace() / "keywords.yaml"


def sparse_index(collection: str) -> Path:
    """
    DEPRECATED: Sparse index is now auto-maintained via PostgreSQL tsvector.

    This function returns the old file path for backwards compatibility
    during migration. It will be removed in a future version.
    """
    warnings.warn(
        "sparse_index() path is deprecated. Sparse index uses PostgreSQL tsvector.",
        DeprecationWarning,
        stacklevel=2,
    )
    return workspace() / "sparse_index" / collection


def ensure_sparse_index_dir() -> Path:
    """
    DEPRECATED: Sparse index is now auto-maintained via PostgreSQL tsvector.

    No longer needed - sparse index is stored in PostgreSQL.
    """
    warnings.warn(
        "ensure_sparse_index_dir() is deprecated. Sparse index uses PostgreSQL tsvector.",
        DeprecationWarning,
        stacklevel=2,
    )
    path = workspace() / "sparse_index"
    path.mkdir(parents=True, exist_ok=True)
    return path


def entity_graph(collection: str) -> Path:
    """
    DEPRECATED: Entity graph is now stored in PostgreSQL.

    This function returns the old SQLite file path for backwards compatibility
    during migration. It will be removed in a future version.
    """
    warnings.warn(
        "entity_graph() path is deprecated. Entity graph is now stored in PostgreSQL.",
        DeprecationWarning,
        stacklevel=2,
    )
    return workspace() / "entity_graph" / f"{collection}.db"


def ensure_entity_graph_dir() -> Path:
    """
    DEPRECATED: Entity graph is now stored in PostgreSQL.

    No longer needed - entity graph is stored in PostgreSQL.
    """
    warnings.warn(
        "ensure_entity_graph_dir() is deprecated. Entity graph is stored in PostgreSQL.",
        DeprecationWarning,
        stacklevel=2,
    )
    path = workspace() / "entity_graph"
    path.mkdir(parents=True, exist_ok=True)
    return path
