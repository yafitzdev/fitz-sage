# fitz_ai/tabular/store/postgres.py
"""
PostgreSQL-based table store with native table support.

Instead of storing compressed CSV blobs, this creates actual PostgreSQL
tables that can be queried directly. This enables:
- Direct SQL queries without loading into memory
- PostgreSQL indexes for fast lookups
- JOIN operations across tables
- Efficient handling of large tables
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import STORAGE
from fitz_ai.storage import get_connection_manager
from fitz_ai.tabular.store.base import StoredTable, compute_hash

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def _sanitize_table_name(table_id: str) -> str:
    """Convert table_id to valid PostgreSQL table name."""
    # Replace non-alphanumeric with underscore
    name = re.sub(r"[^a-zA-Z0-9]", "_", table_id)
    # Ensure starts with letter
    if name and name[0].isdigit():
        name = "t_" + name
    # Truncate to PostgreSQL limit (63 chars) with prefix
    return f"tbl_{name[:55]}".lower()


def _sanitize_column_name(col: str) -> str:
    """Convert column name to valid PostgreSQL identifier."""
    # Replace non-alphanumeric with underscore
    name = re.sub(r"[^a-zA-Z0-9]", "_", col)
    # Ensure starts with letter
    if name and name[0].isdigit():
        name = "c_" + name
    # Handle empty
    if not name:
        name = "col"
    return name.lower()


class PostgresTableStore:
    """
    Table storage using native PostgreSQL tables.

    Creates actual PostgreSQL tables for each stored table, enabling:
    - Direct SQL queries without data loading
    - Proper indexing and query optimization
    - JOIN operations across tables
    - Efficient large table handling

    Schema:
    - `_table_metadata`: Registry of all tables (id, hash, columns, source)
    - `tbl_{table_id}`: Actual data tables with columns as TEXT
    """

    METADATA_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS _table_metadata (
            table_id TEXT PRIMARY KEY,
            table_name TEXT NOT NULL UNIQUE,
            hash TEXT NOT NULL,
            columns TEXT[] NOT NULL,
            column_names_original TEXT[] NOT NULL,
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
        """Create metadata table if not exists."""
        if self._schema_initialized:
            return

        with self._manager.connection(self.collection) as conn:
            conn.execute(self.METADATA_TABLE_SQL)
            conn.commit()

        self._schema_initialized = True
        logger.debug(f"{STORAGE} Table metadata schema initialized for '{self.collection}'")

    def store(
        self,
        table_id: str,
        columns: list[str],
        rows: list[list[str]],
        source_file: str,
    ) -> str:
        """
        Store table as native PostgreSQL table.

        Creates actual table with columns and inserts all rows.

        Args:
            table_id: Unique identifier for the table.
            columns: Column headers (original names).
            rows: Data rows.
            source_file: Original source file path.

        Returns:
            Content hash for cache invalidation.
        """
        self._ensure_schema()

        content_hash = compute_hash(columns, rows)
        table_name = _sanitize_table_name(table_id)
        sanitized_cols = [_sanitize_column_name(c) for c in columns]

        # Handle duplicate column names by appending index
        seen: dict[str, int] = {}
        unique_cols = []
        for col in sanitized_cols:
            if col in seen:
                seen[col] += 1
                unique_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_cols.append(col)
        sanitized_cols = unique_cols

        with self._manager.connection(self.collection) as conn:
            # Drop existing table if exists
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

            # Create table with TEXT columns
            cols_def = ", ".join(f'"{c}" TEXT' for c in sanitized_cols)
            conn.execute(f'CREATE TABLE "{table_name}" ({cols_def})')

            # Insert rows in batches
            if rows:
                placeholders = ", ".join(["%s"] * len(sanitized_cols))
                # Batch insert for performance
                batch_size = 1000
                for i in range(0, len(rows), batch_size):
                    batch = rows[i : i + batch_size]
                    # Pad rows to match column count
                    padded_batch = []
                    for row in batch:
                        if len(row) < len(sanitized_cols):
                            row = row + [""] * (len(sanitized_cols) - len(row))
                        elif len(row) > len(sanitized_cols):
                            row = row[: len(sanitized_cols)]
                        padded_batch.append(tuple(row))

                    conn.executemany(
                        f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                        padded_batch,
                    )

            # Update metadata
            conn.execute(
                """
                INSERT INTO _table_metadata
                    (table_id, table_name, hash, columns, column_names_original, row_count, source_file)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (table_id) DO UPDATE SET
                    table_name = EXCLUDED.table_name,
                    hash = EXCLUDED.hash,
                    columns = EXCLUDED.columns,
                    column_names_original = EXCLUDED.column_names_original,
                    row_count = EXCLUDED.row_count,
                    source_file = EXCLUDED.source_file
                """,
                (table_id, table_name, content_hash, sanitized_cols, columns, len(rows), source_file),
            )
            conn.commit()

        logger.debug(f"{STORAGE} Stored table '{table_id}' as '{table_name}' ({len(rows)} rows)")
        return content_hash

    def retrieve(self, table_id: str) -> StoredTable | None:
        """
        Retrieve table metadata and data.

        Args:
            table_id: Table identifier.

        Returns:
            StoredTable if found, None otherwise.
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            # Get metadata
            result = conn.execute(
                """
                SELECT table_name, hash, columns, column_names_original, row_count, source_file
                FROM _table_metadata
                WHERE table_id = %s
                """,
                (table_id,),
            ).fetchone()

            if not result:
                return None

            table_name, hash_, columns, original_columns, row_count, source_file = result

            # Fetch actual data from table
            try:
                cursor = conn.execute(f'SELECT * FROM "{table_name}"')
                rows = [list(row) for row in cursor.fetchall()]
            except Exception as e:
                logger.warning(f"{STORAGE} Failed to fetch data from '{table_name}': {e}")
                rows = []

            return StoredTable(
                table_id=table_id,
                hash=hash_,
                columns=list(original_columns),  # Return original column names
                rows=rows,
                row_count=row_count,
                source_file=source_file or "",
            )

    def get_table_name(self, table_id: str) -> str | None:
        """
        Get the PostgreSQL table name for a table_id.

        Args:
            table_id: Table identifier.

        Returns:
            PostgreSQL table name if exists, None otherwise.
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT table_name FROM _table_metadata WHERE table_id = %s",
                (table_id,),
            ).fetchone()
            return result[0] if result else None

    def get_columns(self, table_id: str) -> tuple[list[str], list[str]] | None:
        """
        Get column names for a table.

        Args:
            table_id: Table identifier.

        Returns:
            Tuple of (sanitized_columns, original_columns) if exists, None otherwise.
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT columns, column_names_original FROM _table_metadata WHERE table_id = %s",
                (table_id,),
            ).fetchone()
            if result:
                return list(result[0]), list(result[1])
            return None

    def execute_query(
        self,
        table_id: str,
        sql: str,
        params: tuple = (),
    ) -> tuple[list[str], list[list[Any]]] | None:
        """
        Execute SQL query against a table.

        The SQL should reference the table by its sanitized name.
        Use get_table_name() to get the actual table name.

        Args:
            table_id: Table identifier (for connection routing).
            sql: SQL query to execute.
            params: Query parameters.

        Returns:
            Tuple of (column_names, rows) if successful, None on error.
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            try:
                cursor = conn.execute(sql, params)
                col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = [list(row) for row in cursor.fetchall()]
                return col_names, rows
            except Exception as e:
                logger.warning(f"{STORAGE} Query execution failed: {e}")
                return None

    def execute_multi_table_query(
        self,
        sql: str,
        params: tuple = (),
    ) -> tuple[list[str], list[list[Any]]] | None:
        """
        Execute SQL query that may reference multiple tables.

        Args:
            sql: SQL query to execute.
            params: Query parameters.

        Returns:
            Tuple of (column_names, rows) if successful, None on error.
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            try:
                cursor = conn.execute(sql, params)
                col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = [list(row) for row in cursor.fetchall()]
                return col_names, rows
            except Exception as e:
                logger.warning(f"{STORAGE} Multi-table query failed: {e}")
                return None

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
                "SELECT hash FROM _table_metadata WHERE table_id = %s",
                (table_id,),
            ).fetchone()
            return result[0] if result else None

    def get_row_count(self, table_id: str) -> int | None:
        """
        Get row count for a table.

        Args:
            table_id: Table identifier.

        Returns:
            Row count if table exists, None otherwise.
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT row_count FROM _table_metadata WHERE table_id = %s",
                (table_id,),
            ).fetchone()
            return result[0] if result else None

    def list_tables(self) -> list[str]:
        """List all table IDs."""
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            cursor = conn.execute("SELECT table_id FROM _table_metadata ORDER BY table_id")
            return [row[0] for row in cursor]

    def delete(self, table_id: str) -> None:
        """Delete a table and its data."""
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            # Get table name first
            result = conn.execute(
                "SELECT table_name FROM _table_metadata WHERE table_id = %s",
                (table_id,),
            ).fetchone()

            if result:
                table_name = result[0]
                # Drop the actual data table
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                # Remove metadata
                conn.execute("DELETE FROM _table_metadata WHERE table_id = %s", (table_id,))
                conn.commit()
                logger.debug(f"{STORAGE} Deleted table '{table_id}' ('{table_name}')")

    def close(self) -> None:
        """No-op for PostgreSQL (connection pool manages lifecycle)."""
        pass


__all__ = ["PostgresTableStore"]
