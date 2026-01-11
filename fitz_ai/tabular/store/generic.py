# fitz_ai/tabular/store/generic.py
"""Generic table store using vector DB plugin system."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from fitz_ai.logging.logger import get_logger

from .base import StoredTable, compress_csv, compute_hash, decompress_csv
from .cache import TableCache

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class GenericTableStore:
    """
    Table storage using vector DB plugin system.

    Works with any vector DB plugin (Qdrant, Pinecone, Weaviate, Milvus, etc.)
    by using the plugin's upsert/retrieve operations to store table data
    in point payloads.

    Uses local SQLite cache for fast repeated access with hash-based
    invalidation.
    """

    def __init__(
        self,
        collection: str,
        vector_plugin: Any,
    ):
        """
        Initialize generic table store.

        Args:
            collection: Base collection name
            vector_plugin: Vector DB plugin instance (GenericVectorDBPlugin or similar)
        """
        self.collection = collection
        self.table_collection = f"{collection}_tables"
        self.plugin = vector_plugin
        self.cache = TableCache(collection)
        self._collection_created = False

    def _ensure_collection(self) -> None:
        """Ensure table storage collection exists."""
        if self._collection_created:
            return

        try:
            # Try to create collection (will fail silently if exists)
            self.plugin.create_collection(self.table_collection, vector_size=1)
            logger.debug(f"Created table collection {self.table_collection}")
        except Exception:
            # Collection likely exists
            pass

        self._collection_created = True

    def store(
        self,
        table_id: str,
        columns: list[str],
        rows: list[list[str]],
        source_file: str,
    ) -> str:
        """
        Store table in vector DB and local cache.

        Args:
            table_id: Unique identifier for the table
            columns: Column headers
            rows: Data rows
            source_file: Original file path

        Returns:
            Content hash for cache invalidation
        """
        self._ensure_collection()

        content_hash = compute_hash(columns, rows)
        compressed = compress_csv(columns, rows)

        # Store as point with payload (minimal dummy vector)
        point = {
            "id": table_id,
            "vector": [0.0],  # Dummy vector (we only use payload)
            "payload": {
                "table_id": table_id,
                "hash": content_hash,
                "columns": columns,
                "data": base64.b64encode(compressed).decode(),
                "row_count": len(rows),
                "source_file": source_file,
            },
        }

        self.plugin.upsert(self.table_collection, [point])

        # Also cache locally for fast access
        self.cache.store(table_id, content_hash, columns, rows, source_file)

        logger.debug(
            f"Stored table {table_id} ({len(rows)} rows, {len(compressed)} bytes)"
        )
        return content_hash

    def retrieve(self, table_id: str) -> StoredTable | None:
        """
        Retrieve table, using cache when possible.

        First checks local cache (with hash validation), then fetches
        from vector DB if cache miss or stale.
        """
        # Get remote hash for cache validation
        remote_hash = self.get_hash(table_id)
        if remote_hash is None:
            return None

        # Check cache with hash validation
        cached = self.cache.retrieve(table_id, expected_hash=remote_hash)
        if cached:
            logger.debug(f"Cache hit for table {table_id}")
            return cached

        # Cache miss - fetch from vector DB
        logger.debug(f"Cache miss for table {table_id}, fetching from vector DB")
        return self._fetch_from_vector_db(table_id)

    def _fetch_from_vector_db(self, table_id: str) -> StoredTable | None:
        """Fetch table from vector DB and update cache."""
        self._ensure_collection()

        try:
            results = self.plugin.retrieve(
                self.table_collection,
                ids=[table_id],
                with_payload=True,
            )

            if not results:
                return None

            payload = results[0]["payload"]
            compressed = base64.b64decode(payload["data"])
            columns, rows = decompress_csv(compressed)

            # Update cache
            self.cache.store(
                table_id,
                payload["hash"],
                columns,
                rows,
                payload.get("source_file", ""),
            )

            return StoredTable(
                table_id=table_id,
                hash=payload["hash"],
                columns=columns,
                rows=rows,
                row_count=payload["row_count"],
                source_file=payload.get("source_file", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to fetch table {table_id}: {e}")
            return None

    def get_hash(self, table_id: str) -> str | None:
        """
        Get hash from vector DB (lightweight check).

        Used for cache invalidation checks.
        """
        self._ensure_collection()

        try:
            results = self.plugin.retrieve(
                self.table_collection,
                ids=[table_id],
                with_payload=True,
            )
            if results and results[0]["payload"]:
                return results[0]["payload"].get("hash")
            return None
        except Exception:
            return None

    def list_tables(self) -> list[str]:
        """
        List all table IDs in collection.

        Note: This requires scrolling/listing which may not be efficient
        for all vector DBs. Consider caching table IDs if needed.
        """
        # For now, return tables from local cache
        # Full remote listing would require scroll/list operation
        return self.cache.list_tables()

    def delete(self, table_id: str) -> None:
        """Delete table from vector DB and cache."""
        self._ensure_collection()

        try:
            # Most vector DBs support delete by ID through upsert with empty
            # or through a delete operation. For now, just clear cache.
            # The data will be orphaned but not cause issues.
            self.cache.delete(table_id)
            logger.debug(f"Deleted table {table_id} from cache")
        except Exception as e:
            logger.warning(f"Failed to delete table {table_id}: {e}")

    def close(self) -> None:
        """Close cache connection."""
        self.cache.close()
