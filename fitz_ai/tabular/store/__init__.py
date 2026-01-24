# fitz_ai/tabular/store/__init__.py
"""Table storage backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fitz_ai.tabular.store.base import (
    StoredTable,
    TableStore,
    compress_csv,
    compute_hash,
    decompress_csv,
)
from fitz_ai.tabular.store.postgres import PostgresTableStore

if TYPE_CHECKING:
    pass

__all__ = [
    "StoredTable",
    "TableStore",
    "PostgresTableStore",
    "compress_csv",
    "compute_hash",
    "decompress_csv",
    "get_table_store",
]

# pgvector plugins use PostgreSQL table store
PGVECTOR_PLUGINS = {"pgvector", "local-pgvector"}


def get_table_store(
    collection: str,
    vector_db_plugin: str = "pgvector",
    vector_plugin_instance: Any = None,
) -> TableStore:
    """
    Get appropriate table store based on vector DB plugin.

    Args:
        collection: Collection name
        vector_db_plugin: Name of vector DB plugin being used
        vector_plugin_instance: Vector DB plugin instance (unused for pgvector)

    Returns:
        PostgresTableStore for pgvector (unified storage)
    """
    # pgvector: use PostgreSQL table store (unified storage)
    if vector_db_plugin in PGVECTOR_PLUGINS:
        return PostgresTableStore(collection)

    # Default to PostgreSQL
    return PostgresTableStore(collection)
