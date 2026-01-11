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
from fitz_ai.tabular.store.sqlite import SqliteTableStore

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

__all__ = [
    "StoredTable",
    "TableStore",
    "SqliteTableStore",
    "compress_csv",
    "compute_hash",
    "decompress_csv",
    "get_table_store",
]


def get_table_store(
    collection: str,
    vector_db_plugin: str = "local_faiss",
    qdrant_client: "QdrantClient | None" = None,
) -> TableStore:
    """
    Get appropriate table store based on vector DB plugin.

    Args:
        collection: Collection name
        vector_db_plugin: Name of vector DB plugin being used
        qdrant_client: Qdrant client instance (required for qdrant plugin)

    Returns:
        TableStore instance (SqliteTableStore for local, QdrantTableStore for team)
    """
    if vector_db_plugin == "qdrant":
        if qdrant_client is None:
            raise ValueError("Qdrant client required for qdrant table store")

        from fitz_ai.tabular.store.qdrant import QdrantTableStore

        return QdrantTableStore(collection, qdrant_client)

    # Default to local SQLite for all other plugins
    return SqliteTableStore(collection)
