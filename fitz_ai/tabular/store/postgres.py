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
            file_hash TEXT,
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
        file_hash: str | None = None,
    ) -> str:
        """
        Store table as native PostgreSQL table.

        Creates actual table with columns and inserts all rows.
        Includes _row_num column for incremental column addition.

        Args:
            table_id: Unique identifier for the table.
            columns: Column headers (original names).
            rows: Data rows.
            source_file: Original source file path.
            file_hash: Hash of source file for change detection.

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

            # Create table with _row_num + TEXT columns
            cols_def = "_row_num INTEGER PRIMARY KEY, " + ", ".join(
                f'"{c}" TEXT' for c in sanitized_cols
            )
            conn.execute(f'CREATE TABLE "{table_name}" ({cols_def})')

            # Insert rows in batches with row numbers
            if rows:
                placeholders = ", ".join(["%s"] * (len(sanitized_cols) + 1))  # +1 for _row_num
                # Batch insert for performance
                batch_size = 1000
                with conn.cursor() as cur:
                    for batch_start in range(0, len(rows), batch_size):
                        batch = rows[batch_start : batch_start + batch_size]
                        # Pad rows to match column count and prepend row number
                        padded_batch = []
                        for i, row in enumerate(batch):
                            row_num = batch_start + i
                            if len(row) < len(sanitized_cols):
                                row = row + [""] * (len(sanitized_cols) - len(row))
                            elif len(row) > len(sanitized_cols):
                                row = row[: len(sanitized_cols)]
                            padded_batch.append((row_num, *row))

                        cur.executemany(
                            f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                            padded_batch,
                        )

            # Update metadata
            conn.execute(
                """
                INSERT INTO _table_metadata
                    (table_id, table_name, hash, columns, column_names_original,
                     row_count, source_file, file_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (table_id) DO UPDATE SET
                    table_name = EXCLUDED.table_name,
                    hash = EXCLUDED.hash,
                    columns = EXCLUDED.columns,
                    column_names_original = EXCLUDED.column_names_original,
                    row_count = EXCLUDED.row_count,
                    source_file = EXCLUDED.source_file,
                    file_hash = EXCLUDED.file_hash
                """,
                (table_id, table_name, content_hash, sanitized_cols, columns,
                 len(rows), source_file, file_hash),
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

            # Fetch actual data from table (excluding _row_num)
            try:
                cols_str = ", ".join(f'"{c}"' for c in columns)
                cursor = conn.execute(
                    f'SELECT {cols_str} FROM "{table_name}" ORDER BY _row_num'
                )
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
                # If no params, escape % chars to avoid placeholder interpretation
                if not params:
                    sql = sql.replace("%", "%%")
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
                # If no params, escape % chars to avoid placeholder interpretation
                if not params:
                    sql = sql.replace("%", "%%")
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

    def get_file_hash(self, table_id: str) -> str | None:
        """
        Get file hash for change detection.

        Args:
            table_id: Table identifier.

        Returns:
            File hash if exists, None otherwise.
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT file_hash FROM _table_metadata WHERE table_id = %s",
                (table_id,),
            ).fetchone()
            return result[0] if result else None

    def add_columns(
        self,
        table_id: str,
        new_columns: list[str],
        column_values: list[list[str]],
    ) -> bool:
        """
        Add new columns to an existing table.

        Uses _row_num for row alignment.

        Args:
            table_id: Table identifier.
            new_columns: Original column names to add.
            column_values: Values for each row, aligned by index.
                          column_values[row_idx][col_idx]

        Returns:
            True if successful, False otherwise.
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            # Get table metadata
            result = conn.execute(
                """
                SELECT table_name, columns, column_names_original
                FROM _table_metadata WHERE table_id = %s
                """,
                (table_id,),
            ).fetchone()

            if not result:
                logger.warning(f"{STORAGE} Table '{table_id}' not found for column addition")
                return False

            table_name, existing_cols, existing_original = result
            existing_cols = list(existing_cols)
            existing_original = list(existing_original)

            # Sanitize new column names
            sanitized_new = [_sanitize_column_name(c) for c in new_columns]

            # Handle duplicates with existing columns
            for i, col in enumerate(sanitized_new):
                if col in existing_cols:
                    # Append suffix to make unique
                    suffix = 1
                    while f"{col}_{suffix}" in existing_cols:
                        suffix += 1
                    sanitized_new[i] = f"{col}_{suffix}"

            try:
                # Add columns to table
                for col in sanitized_new:
                    conn.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" TEXT')

                # Update values row by row using _row_num
                for row_num, row_values in enumerate(column_values):
                    if not row_values:
                        continue
                    set_clause = ", ".join(
                        f'"{col}" = %s' for col in sanitized_new[: len(row_values)]
                    )
                    conn.execute(
                        f'UPDATE "{table_name}" SET {set_clause} WHERE _row_num = %s',
                        (*row_values[: len(sanitized_new)], row_num),
                    )

                # Update metadata
                updated_cols = existing_cols + sanitized_new
                updated_original = existing_original + new_columns
                conn.execute(
                    """
                    UPDATE _table_metadata
                    SET columns = %s, column_names_original = %s
                    WHERE table_id = %s
                    """,
                    (updated_cols, updated_original, table_id),
                )
                conn.commit()

                logger.debug(
                    f"{STORAGE} Added {len(new_columns)} columns to '{table_id}': {new_columns}"
                )
                return True

            except Exception as e:
                logger.warning(f"{STORAGE} Failed to add columns to '{table_id}': {e}")
                conn.rollback()
                return False

    def has_columns(self, table_id: str, columns: list[str]) -> tuple[list[str], list[str]]:
        """
        Check which columns exist in a table.

        Args:
            table_id: Table identifier.
            columns: Original column names to check.

        Returns:
            Tuple of (existing_columns, missing_columns).
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT column_names_original FROM _table_metadata WHERE table_id = %s",
                (table_id,),
            ).fetchone()

            if not result:
                return [], columns

            existing_original = set(result[0])
            existing = [c for c in columns if c in existing_original]
            missing = [c for c in columns if c not in existing_original]
            return existing, missing

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
