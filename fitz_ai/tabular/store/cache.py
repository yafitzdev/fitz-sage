# fitz_ai/tabular/store/cache.py
"""Local SQLite cache for team mode table data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

from .base import StoredTable, compress_csv, decompress_csv

logger = get_logger(__name__)


class TableCache:
    """
    Local cache for Qdrant table data.

    Caches table data from Qdrant in local SQLite for fast repeated access.
    Uses hash-based invalidation to detect stale cache entries.
    """

    def __init__(self, collection: str):
        self.collection = collection
        self._conn: sqlite3.Connection | None = None

    @property
    def cache_path(self) -> Path:
        """Path to cache database file."""
        return FitzPaths.cache() / "tables" / f"{self.collection}.db"

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.cache_path))
            self._ensure_schema()
        return self._conn

    def _ensure_schema(self) -> None:
        """Create cache schema if not exists."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                table_id TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                columns TEXT NOT NULL,
                data BLOB NOT NULL,
                row_count INTEGER NOT NULL,
                source_file TEXT
            )
        """)
        self.conn.commit()

    def store(
        self,
        table_id: str,
        hash: str,
        columns: list[str],
        rows: list[list[str]],
        source_file: str = "",
    ) -> None:
        """
        Store table in cache.

        Args:
            table_id: Table identifier
            hash: Content hash for invalidation
            columns: Column headers
            rows: Data rows
            source_file: Original file path
        """
        compressed = compress_csv(columns, rows)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO cache
            (table_id, hash, columns, data, row_count, source_file)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (table_id, hash, ",".join(columns), compressed, len(rows), source_file),
        )
        self.conn.commit()
        logger.debug(f"Cached table {table_id}")

    def retrieve(self, table_id: str, expected_hash: str) -> StoredTable | None:
        """
        Retrieve from cache if hash matches.

        Args:
            table_id: Table identifier
            expected_hash: Expected content hash (from remote)

        Returns:
            StoredTable if cache hit and hash matches, None otherwise
        """
        cursor = self.conn.execute(
            "SELECT hash, columns, data, row_count, source_file FROM cache WHERE table_id = ?",
            (table_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        cached_hash, columns_str, data, row_count, source_file = row

        # Hash mismatch - cache is stale
        if cached_hash != expected_hash:
            logger.debug(f"Cache stale for {table_id}: {cached_hash} != {expected_hash}")
            return None

        columns, rows = decompress_csv(data)
        return StoredTable(
            table_id=table_id,
            hash=cached_hash,
            columns=columns,
            rows=rows,
            row_count=row_count,
            source_file=source_file or "",
        )

    def get_cached_hash(self, table_id: str) -> str | None:
        """Get cached hash for a table."""
        cursor = self.conn.execute(
            "SELECT hash FROM cache WHERE table_id = ?",
            (table_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def delete(self, table_id: str) -> None:
        """Remove table from cache."""
        self.conn.execute("DELETE FROM cache WHERE table_id = ?", (table_id,))
        self.conn.commit()

    def clear(self) -> None:
        """Clear entire cache."""
        self.conn.execute("DELETE FROM cache")
        self.conn.commit()
        logger.debug(f"Cleared table cache for {self.collection}")

    def list_tables(self) -> list[str]:
        """List all table IDs in cache."""
        cursor = self.conn.execute("SELECT table_id FROM cache")
        return [row[0] for row in cursor.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
