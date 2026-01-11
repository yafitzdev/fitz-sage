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
from fitz_ai.tabular.store.generic import GenericTableStore
from fitz_ai.tabular.store.sqlite import SqliteTableStore

if TYPE_CHECKING:
    pass

__all__ = [
    "StoredTable",
    "TableStore",
    "SqliteTableStore",
    "GenericTableStore",
    "compress_csv",
    "compute_hash",
    "decompress_csv",
    "get_table_store",
]

# Local plugins that use SQLite table store (no remote sync)
LOCAL_PLUGINS = {"local_faiss", "local-faiss"}


def get_table_store(
    collection: str,
    vector_db_plugin: str = "local_faiss",
    vector_plugin_instance: Any = None,
) -> TableStore:
    """
    Get appropriate table store based on vector DB plugin.

    Args:
        collection: Collection name
        vector_db_plugin: Name of vector DB plugin being used
        vector_plugin_instance: Vector DB plugin instance (for remote stores)

    Returns:
        TableStore instance:
        - SqliteTableStore for local plugins (FAISS)
        - GenericTableStore for remote plugins (Qdrant, Pinecone, Weaviate, Milvus)
    """
    # Local mode: use SQLite directly
    if vector_db_plugin in LOCAL_PLUGINS:
        return SqliteTableStore(collection)

    # Remote mode: use GenericTableStore with plugin's retrieve operation
    if vector_plugin_instance is not None:
        return GenericTableStore(collection, vector_plugin_instance)

    # Fallback to SQLite if no plugin instance provided
    return SqliteTableStore(collection)
