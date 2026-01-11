# fitz_ai/tabular/store/sqlite.py
"""SQLite-based table store for local mode."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

from .base import StoredTable, compress_csv, compute_hash, decompress_csv

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class SqliteTableStore:
    """
    Local table storage using SQLite.

    Stores compressed CSV data in a SQLite database for zero-dependency
    table persistence. Used in local mode where no shared access is needed.
    """

    def __init__(self, collection: str):
        self.collection = collection
        self._conn: sqlite3.Connection | None = None

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database file."""
        return FitzPaths.workspace() / "tables" / f"{self.collection}.db"

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._ensure_schema()
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables schema if not exists."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tables (
                table_id TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                columns TEXT NOT NULL,
                data BLOB NOT NULL,
                row_count INTEGER NOT NULL,
                source_file TEXT
            )
        """
        )
        self.conn.commit()

    def store(
        self,
        table_id: str,
        columns: list[str],
        rows: list[list[str]],
        source_file: str,
    ) -> str:
        """
        Store table data with compression.

        Args:
            table_id: Unique identifier for the table
            columns: Column headers
            rows: Data rows
            source_file: Original file path

        Returns:
            Content hash for cache invalidation
        """
        content_hash = compute_hash(columns, rows)
        compressed = compress_csv(columns, rows)

        self.conn.execute(
            """
            INSERT OR REPLACE INTO tables
            (table_id, hash, columns, data, row_count, source_file)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                table_id,
                content_hash,
                ",".join(columns),
                compressed,
                len(rows),
                source_file,
            ),
        )
        self.conn.commit()

        logger.debug(
            f"Stored table {table_id} ({len(rows)} rows, {len(compressed)} bytes compressed)"
        )
        return content_hash

    def retrieve(self, table_id: str) -> StoredTable | None:
        """
        Retrieve table by ID.

        Args:
            table_id: Table identifier

        Returns:
            StoredTable if found, None otherwise
        """
        cursor = self.conn.execute(
            "SELECT hash, columns, data, row_count, source_file FROM tables WHERE table_id = ?",
            (table_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        hash_, columns_str, data, row_count, source_file = row
        columns, rows = decompress_csv(data)

        return StoredTable(
            table_id=table_id,
            hash=hash_,
            columns=columns,
            rows=rows,
            row_count=row_count,
            source_file=source_file or "",
        )

    def get_hash(self, table_id: str) -> str | None:
        """
        Get hash without retrieving full data.

        Useful for cache invalidation checks.
        """
        cursor = self.conn.execute(
            "SELECT hash FROM tables WHERE table_id = ?",
            (table_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def list_tables(self) -> list[str]:
        """List all stored table IDs."""
        cursor = self.conn.execute("SELECT table_id FROM tables")
        return [row[0] for row in cursor.fetchall()]

    def delete(self, table_id: str) -> None:
        """Delete a table by ID."""
        self.conn.execute("DELETE FROM tables WHERE table_id = ?", (table_id,))
        self.conn.commit()
        logger.debug(f"Deleted table {table_id}")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
