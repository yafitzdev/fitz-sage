# fitz_ai/structured/ingestion.py
"""
Structured data ingestion pipeline.

Ingests tables/CSVs into vector DB with schema registration
and row storage for metadata filtering.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from fitz_ai.logging.logger import get_logger
from fitz_ai.structured.constants import (
    FIELD_PRIMARY_KEY,
    FIELD_ROW_DATA,
    FIELD_TABLE,
    MAX_INDEXED_COLUMNS,
    MAX_SCAN_ROWS,
    UPSERT_BATCH_SIZE,
    get_tables_collection,
)
from fitz_ai.structured.schema import ColumnSchema, SchemaStore, TableSchema
from fitz_ai.structured.types import (
    coerce_value,
    infer_column_type,
    select_indexed_columns,
)

logger = get_logger(__name__)


class TableTooLargeError(Exception):
    """Raised when table exceeds MAX_SCAN_ROWS limit."""

    def __init__(self, table_name: str, row_count: int, max_rows: int = MAX_SCAN_ROWS):
        self.table_name = table_name
        self.row_count = row_count
        self.max_rows = max_rows
        super().__init__(
            f"Table '{table_name}' has {row_count} rows, exceeding limit of {max_rows}. "
            f"Use a database for large tables."
        )


class MissingPrimaryKeyError(Exception):
    """Raised when primary key column is missing from rows."""

    def __init__(self, table_name: str, primary_key: str):
        self.table_name = table_name
        self.primary_key = primary_key
        super().__init__(
            f"Primary key column '{primary_key}' not found in table '{table_name}'"
        )


@runtime_checkable
class VectorDBClient(Protocol):
    """Protocol for vector DB operations."""

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Upsert points to collection."""
        ...

    def delete_collection(self, collection: str) -> int:
        """Delete collection."""
        ...


class StructuredIngester:
    """
    Ingests structured data (tables/CSVs) into vector DB.

    Stores:
    - Schema in {collection}__schema for discovery
    - Rows in {collection}__tables for metadata filtering
    """

    def __init__(
        self,
        vector_db: VectorDBClient,
        schema_store: SchemaStore,
        base_collection: str,
        vector_dim: int = 1536,
    ):
        """
        Initialize ingester.

        Args:
            vector_db: Vector DB client for row storage
            schema_store: Schema store for registration
            base_collection: Base collection name
            vector_dim: Dimension for zero vectors (must match embedding dim)
        """
        self._vector_db = vector_db
        self._schema_store = schema_store
        self._base_collection = base_collection
        self._tables_collection = get_tables_collection(base_collection)
        self._vector_dim = vector_dim

    @property
    def tables_collection(self) -> str:
        """Get the tables collection name."""
        return self._tables_collection

    def ingest_table(
        self,
        table_name: str,
        rows: list[dict[str, Any]],
        primary_key: str,
        indexed_columns: list[str] | None = None,
        replace: bool = True,
    ) -> TableSchema:
        """
        Ingest a table into the vector DB.

        Args:
            table_name: Name of the table
            rows: List of row dictionaries
            primary_key: Name of the primary key column
            indexed_columns: Columns to index (auto-selected if None)
            replace: If True, delete existing table data first

        Returns:
            The registered TableSchema

        Raises:
            TableTooLargeError: If table exceeds MAX_SCAN_ROWS
            MissingPrimaryKeyError: If primary key column missing
        """
        # Validate size
        if len(rows) > MAX_SCAN_ROWS:
            raise TableTooLargeError(table_name, len(rows))

        if not rows:
            logger.warning(f"Empty table '{table_name}', skipping ingestion")
            return TableSchema(
                table_name=table_name,
                columns=[],
                primary_key=primary_key,
                row_count=0,
            )

        # Validate primary key exists
        if primary_key not in rows[0]:
            raise MissingPrimaryKeyError(table_name, primary_key)

        # Infer schema from rows
        column_names = list(rows[0].keys())
        column_types = self._infer_types(column_names, rows)

        # Auto-select indexed columns if not provided
        if indexed_columns is None:
            sample_values = self._extract_column_samples(column_names, rows)
            indexed_columns = select_indexed_columns(
                column_names,
                column_types,
                sample_values,
                primary_key,
                max_indexed=MAX_INDEXED_COLUMNS,
            )

        # Build schema
        columns = [
            ColumnSchema(
                name=name,
                type=col_type,
                nullable=True,
                indexed=(name in indexed_columns),
            )
            for name, col_type in zip(column_names, column_types)
        ]

        schema = TableSchema(
            table_name=table_name,
            columns=columns,
            primary_key=primary_key,
            row_count=len(rows),
        )

        # Delete existing data if replacing
        if replace:
            self._delete_table_rows(table_name)

        # Store rows in batches
        self._store_rows(table_name, rows, schema)

        # Register schema (after successful row storage)
        self._schema_store.register_table(schema)

        logger.info(
            f"Ingested table '{table_name}': {len(rows)} rows, "
            f"{len(columns)} columns, indexed: {indexed_columns}"
        )

        return schema

    def delete_table(self, table_name: str) -> int:
        """
        Delete a table and its schema.

        Args:
            table_name: Name of the table to delete

        Returns:
            Number of rows deleted
        """
        deleted = self._delete_table_rows(table_name)
        self._schema_store.unregister_table(table_name)
        logger.info(f"Deleted table '{table_name}': {deleted} rows")
        return deleted

    def _infer_types(
        self, column_names: list[str], rows: list[dict[str, Any]]
    ) -> list[str]:
        """Infer column types from sample rows."""
        types = []
        for col_name in column_names:
            values = [row.get(col_name) for row in rows[:100]]  # Sample first 100
            col_type = infer_column_type(values)
            types.append(col_type)
        return types

    def _extract_column_samples(
        self, column_names: list[str], rows: list[dict[str, Any]]
    ) -> list[list[Any]]:
        """Extract sample values for each column."""
        samples = []
        for col_name in column_names:
            values = [row.get(col_name) for row in rows[:100]]
            samples.append(values)
        return samples

    def _store_rows(
        self,
        table_name: str,
        rows: list[dict[str, Any]],
        schema: TableSchema,
    ) -> None:
        """Store rows in the tables collection."""
        # Get indexed column names for payload extraction
        indexed_cols = schema.indexed_columns
        pk_col = schema.primary_key

        # Create zero vector
        zero_vector = [0.0] * self._vector_dim

        # Process in batches
        for batch_start in range(0, len(rows), UPSERT_BATCH_SIZE):
            batch_end = min(batch_start + UPSERT_BATCH_SIZE, len(rows))
            batch = rows[batch_start:batch_end]

            points = []
            for row in batch:
                pk_value = str(row.get(pk_col, ""))
                if not pk_value:
                    logger.warning(f"Skipping row with empty primary key in {table_name}")
                    continue

                # Build payload with indexed columns at top level
                payload: dict[str, Any] = {
                    FIELD_TABLE: table_name,
                    FIELD_PRIMARY_KEY: pk_value,
                }

                # Add indexed columns at top level (for filtering)
                for col_name in indexed_cols:
                    if col_name in row:
                        col_schema = schema.get_column(col_name)
                        if col_schema:
                            payload[col_name] = coerce_value(
                                row[col_name], col_schema.type
                            )

                # Store full row data
                payload[FIELD_ROW_DATA] = row

                point = {
                    "id": f"{table_name}:{pk_value}",
                    "vector": zero_vector,
                    "payload": payload,
                }
                points.append(point)

            if points:
                self._vector_db.upsert(self._tables_collection, points)

    def _delete_table_rows(self, table_name: str) -> int:
        """
        Delete all rows for a table.

        Note: This is a placeholder - proper implementation requires
        vector DB support for filtered deletion.
        """
        # TODO: Implement proper deletion when VDB supports filtered delete
        # For now, this is a no-op - rows will be overwritten on re-ingest
        logger.debug(f"Delete rows for table '{table_name}' (not implemented)")
        return 0


__all__ = [
    "StructuredIngester",
    "TableTooLargeError",
    "MissingPrimaryKeyError",
]
