# fitz_ai/storage/__init__.py
"""
Unified PostgreSQL storage for fitz-ai.

This module provides the storage layer that replaces FAISS + SQLite
with PostgreSQL + pgvector for unified vector and tabular storage.

Usage:
    from fitz_ai.storage import get_connection_manager, get_connection

    # Get connection manager singleton
    manager = get_connection_manager()
    manager.start()

    # Get connection for a collection
    with get_connection("my_collection") as conn:
        conn.execute("SELECT * FROM chunks LIMIT 10")
"""

from fitz_ai.storage.config import StorageConfig, StorageMode
from fitz_ai.storage.postgres import (
    PostgresConnectionManager,
    get_connection,
    get_connection_manager,
)

__all__ = [
    "StorageConfig",
    "StorageMode",
    "PostgresConnectionManager",
    "get_connection_manager",
    "get_connection",
]
