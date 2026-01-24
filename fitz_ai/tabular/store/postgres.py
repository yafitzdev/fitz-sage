# fitz_ai/tabular/store/postgres.py
"""
PostgreSQL-based table store for unified storage.

Replaces SqliteTableStore for pgvector deployments.
Uses same database as vectors for unified storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import STORAGE
from fitz_ai.storage import get_connection_manager
from fitz_ai.tabular.store.base import (
    StoredTable,
    compress_csv,
    compute_hash,
    decompress_csv,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class PostgresTableStore:
    """
    Table storage using PostgreSQL.

    Replaces SqliteTableStore for pgvector deployments.
    Uses same database as vectors for unified storage.

    Features:
    - Compressed storage (gzipped CSV in BYTEA column)
    - Hash-based change detection for cache invalidation
    - Full SQL query support
    """

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS tables (
            table_id TEXT PRIMARY KEY,
            hash TEXT NOT NULL,
            columns TEXT[] NOT NULL,
            data BYTEA NOT NULL,
            row_count INTEGER NOT NULL,
            source_file TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """

    def __init__(self, collection: str):
        """
        Initialize PostgreSQL table store.

        Args:
            collection: Collection name (maps to database).
        """
        self.collection = collection
        self._manager = get_connection_manager()
        self._manager.start()
        self._schema_initialized = False

    def _ensure_schema(self) -> None:
        """Create tables schema if not exists."""
        if self._schema_initialized:
            return

        with self._manager.connection(self.collection) as conn:
            conn.execute(self.CREATE_TABLE_SQL)
            conn.commit()

        self._schema_initialized = True
        logger.debug(f"{STORAGE} Tables schema initialized for '{self.collection}'")

    def store(
        self,
        table_id: str,
        columns: list[str],
        rows: list[list[str]],
        source_file: str,
    ) -> str:
        """
        Store table with compression.

        Args:
            table_id: Unique identifier for the table.
            columns: Column headers.
            rows: Data rows.
            source_file: Original source file path.

        Returns:
            Content hash for cache invalidation.
        """
        self._ensure_schema()

        content_hash = compute_hash(columns, rows)
        compressed = compress_csv(columns, rows)

        with self._manager.connection(self.collection) as conn:
            conn.execute(
                """
                INSERT INTO tables (table_id, hash, columns, data, row_count, source_file)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (table_id) DO UPDATE SET
                    hash = EXCLUDED.hash,
                    columns = EXCLUDED.columns,
                    data = EXCLUDED.data,
                    row_count = EXCLUDED.row_count,
                    source_file = EXCLUDED.source_file
                """,
                (table_id, content_hash, columns, compressed, len(rows), source_file),
            )
            conn.commit()

        logger.debug(f"{STORAGE} Stored table '{table_id}' ({len(rows)} rows)")
        return content_hash

    def retrieve(self, table_id: str) -> StoredTable | None:
        """
        Retrieve table by ID.

        Args:
            table_id: Table identifier.

        Returns:
            StoredTable if found, None otherwise.
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                """
                SELECT hash, columns, data, row_count, source_file
                FROM tables
                WHERE table_id = %s
                """,
                (table_id,),
            ).fetchone()

            if not result:
                return None

            hash_, columns, data, row_count, source_file = result

            # Decompress data
            _, rows = decompress_csv(bytes(data))

            return StoredTable(
                table_id=table_id,
                hash=hash_,
                columns=list(columns),
                rows=rows,
                row_count=row_count,
                source_file=source_file or "",
            )

    def get_hash(self, table_id: str) -> str | None:
        """
        Get hash without full data retrieval (for cache check).

        Args:
            table_id: Table identifier.

        Returns:
            Hash string if table exists, None otherwise.
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT hash FROM tables WHERE table_id = %s",
                (table_id,),
            ).fetchone()
            return result[0] if result else None

    def list_tables(self) -> list[str]:
        """List all table IDs."""
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            cursor = conn.execute("SELECT table_id FROM tables ORDER BY table_id")
            return [row[0] for row in cursor]

    def delete(self, table_id: str) -> None:
        """Delete a table."""
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            conn.execute("DELETE FROM tables WHERE table_id = %s", (table_id,))
            conn.commit()

        logger.debug(f"{STORAGE} Deleted table '{table_id}'")

    def close(self) -> None:
        """No-op for PostgreSQL (connection pool manages lifecycle)."""
        pass


__all__ = ["PostgresTableStore"]
