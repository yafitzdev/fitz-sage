# fitz_ai/tabular/store/qdrant.py
"""Qdrant-based table store for team mode."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from fitz_ai.logging.logger import get_logger

from .base import StoredTable, compress_csv, compute_hash, decompress_csv
from .cache import TableCache

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = get_logger(__name__)


class QdrantTableStore:
    """
    Team table storage using Qdrant payloads.

    Stores compressed CSV data in Qdrant point payloads in a separate
    collection from vectors. Uses local SQLite cache for fast repeated access.
    """

    def __init__(
        self,
        collection: str,
        client: "QdrantClient",
    ):
        self.collection = collection
        self.table_collection = f"{collection}_tables"
        self.client = client
        self.cache = TableCache(collection)
        self._collection_exists = False

    def _ensure_collection(self) -> None:
        """Ensure table storage collection exists."""
        if self._collection_exists:
            return

        try:
            from qdrant_client.models import Distance, VectorParams

            # Check if collection exists
            collections = self.client.get_collections().collections
            if not any(c.name == self.table_collection for c in collections):
                # Create collection with minimal vector config (we only use payloads)
                self.client.create_collection(
                    collection_name=self.table_collection,
                    vectors_config=VectorParams(size=1, distance=Distance.COSINE),
                )
                logger.debug(f"Created table collection {self.table_collection}")

            self._collection_exists = True
        except Exception as e:
            logger.warning(f"Failed to ensure table collection: {e}")
            raise

    def store(
        self,
        table_id: str,
        columns: list[str],
        rows: list[list[str]],
        source_file: str,
    ) -> str:
        """
        Store table in Qdrant and local cache.

        Args:
            table_id: Unique identifier for the table
            columns: Column headers
            rows: Data rows
            source_file: Original file path

        Returns:
            Content hash for cache invalidation
        """
        from qdrant_client.models import PointStruct

        self._ensure_collection()

        content_hash = compute_hash(columns, rows)
        compressed = compress_csv(columns, rows)

        # Store in Qdrant as base64 (payloads don't accept raw bytes)
        self.client.upsert(
            collection_name=self.table_collection,
            points=[
                PointStruct(
                    id=self._table_id_to_point_id(table_id),
                    vector=[0.0],  # Dummy vector (we only use payload)
                    payload={
                        "table_id": table_id,
                        "hash": content_hash,
                        "columns": columns,
                        "data": base64.b64encode(compressed).decode(),
                        "row_count": len(rows),
                        "source_file": source_file,
                    },
                )
            ],
        )

        # Also cache locally for fast access
        self.cache.store(table_id, content_hash, columns, rows, source_file)

        logger.debug(
            f"Stored table {table_id} in Qdrant ({len(rows)} rows, {len(compressed)} bytes)"
        )
        return content_hash

    def retrieve(self, table_id: str) -> StoredTable | None:
        """
        Retrieve table, using cache when possible.

        First checks local cache (with hash validation), then fetches
        from Qdrant if cache miss or stale.
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

        # Cache miss - fetch from Qdrant
        logger.debug(f"Cache miss for table {table_id}, fetching from Qdrant")
        return self._fetch_from_qdrant(table_id)

    def _fetch_from_qdrant(self, table_id: str) -> StoredTable | None:
        """Fetch table from Qdrant and update cache."""
        self._ensure_collection()

        try:
            result = self.client.retrieve(
                collection_name=self.table_collection,
                ids=[self._table_id_to_point_id(table_id)],
                with_payload=True,
            )

            if not result:
                return None

            payload = result[0].payload
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
        Get hash from Qdrant (lightweight, minimal data transfer).

        Used for cache invalidation checks.
        """
        self._ensure_collection()

        try:
            result = self.client.retrieve(
                collection_name=self.table_collection,
                ids=[self._table_id_to_point_id(table_id)],
                with_payload=["hash"],  # Only fetch hash field
            )
            return result[0].payload["hash"] if result else None
        except Exception:
            return None

    def list_tables(self) -> list[str]:
        """List all table IDs in collection."""
        self._ensure_collection()

        try:
            # Scroll through all points
            result, _ = self.client.scroll(
                collection_name=self.table_collection,
                limit=10000,
                with_payload=["table_id"],
            )
            return [p.payload["table_id"] for p in result]
        except Exception as e:
            logger.warning(f"Failed to list tables: {e}")
            return []

    def delete(self, table_id: str) -> None:
        """Delete table from Qdrant and cache."""
        self._ensure_collection()

        try:
            from qdrant_client.models import PointIdsList

            self.client.delete(
                collection_name=self.table_collection,
                points_selector=PointIdsList(
                    points=[self._table_id_to_point_id(table_id)],
                ),
            )
            self.cache.delete(table_id)
            logger.debug(f"Deleted table {table_id}")
        except Exception as e:
            logger.warning(f"Failed to delete table {table_id}: {e}")

    def _table_id_to_point_id(self, table_id: str) -> int:
        """
        Convert table_id string to Qdrant point ID.

        Qdrant requires numeric IDs, so we hash the string to get
        a consistent integer ID.
        """
        import hashlib

        # Use first 8 bytes of hash as unsigned 64-bit int
        hash_bytes = hashlib.sha256(table_id.encode()).digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big")

    def close(self) -> None:
        """Close cache connection."""
        self.cache.close()
