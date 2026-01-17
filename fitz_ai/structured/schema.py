# fitz_ai/structured/schema.py
"""
Schema storage and discovery for structured data.

Stores table schemas in a dedicated vector DB collection for semantic search.
Enables query routing by finding tables relevant to user questions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fitz_ai.logging.logger import get_logger
from fitz_ai.structured.constants import SCHEMA_COLLECTION_SUFFIX

logger = get_logger(__name__)


@dataclass
class ColumnSchema:
    """Schema for a single column."""

    name: str
    type: str  # "string", "number", "date", "boolean"
    nullable: bool = True
    indexed: bool = False  # Whether this column is indexed for filtering


@dataclass
class TableSchema:
    """
    Schema for a structured table.

    Stored in {collection}__schema for discovery via semantic search.
    """

    table_name: str
    columns: list[ColumnSchema]
    primary_key: str
    row_count: int = 0
    version: str = ""  # Hash of table state for invalidation

    def __post_init__(self):
        if not self.version:
            self.version = self._compute_version()

    def _compute_version(self) -> str:
        """Compute version hash from schema + row count."""
        data = {
            "table": self.table_name,
            "columns": [(c.name, c.type) for c in self.columns],
            "pk": self.primary_key,
            "rows": self.row_count,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def column_names(self) -> list[str]:
        """Get list of column names."""
        return [c.name for c in self.columns]

    @property
    def indexed_columns(self) -> list[str]:
        """Get list of indexed column names."""
        return [c.name for c in self.columns if c.indexed]

    def get_column(self, name: str) -> ColumnSchema | None:
        """Get column by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def to_search_text(self) -> str:
        """
        Generate searchable text for semantic matching.

        Includes table name and all column names for discovery.
        """
        col_text = ", ".join(self.column_names)
        return f"{self.table_name}: {col_text}"

    def to_payload(self) -> dict[str, Any]:
        """Convert to vector DB payload format."""
        return {
            "table_name": self.table_name,
            "columns": [
                {
                    "name": c.name,
                    "type": c.type,
                    "nullable": c.nullable,
                    "indexed": c.indexed,
                }
                for c in self.columns
            ],
            "primary_key": self.primary_key,
            "row_count": self.row_count,
            "version": self.version,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TableSchema":
        """Create from vector DB payload."""
        columns = [
            ColumnSchema(
                name=c["name"],
                type=c["type"],
                nullable=c.get("nullable", True),
                indexed=c.get("indexed", False),
            )
            for c in payload["columns"]
        ]
        return cls(
            table_name=payload["table_name"],
            columns=columns,
            primary_key=payload["primary_key"],
            row_count=payload.get("row_count", 0),
            version=payload.get("version", ""),
        )


@dataclass
class SchemaSearchResult:
    """Result from schema search."""

    schema: TableSchema
    score: float


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding generation."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        ...


@runtime_checkable
class VectorDBClient(Protocol):
    """Protocol for vector DB operations."""

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Upsert points to collection."""
        ...

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> list[Any]:
        """Search collection."""
        ...

    def delete_collection(self, collection: str) -> int:
        """Delete collection."""
        ...

    def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve points by ID."""
        ...


class SchemaStore:
    """
    Storage and discovery for table schemas.

    Stores schemas in {collection}__schema with embeddings of
    "table_name: col1, col2, col3" for semantic discovery.
    """

    def __init__(
        self,
        vector_db: VectorDBClient,
        embedding: EmbeddingClient,
        base_collection: str,
    ):
        """
        Initialize schema store.

        Args:
            vector_db: Vector DB client for storage
            embedding: Embedding client for search text
            base_collection: Base collection name (schemas go in {base}__schema)
        """
        self._vector_db = vector_db
        self._embedding = embedding
        self._base_collection = base_collection
        self._schema_collection = f"{base_collection}{SCHEMA_COLLECTION_SUFFIX}"

    @property
    def schema_collection(self) -> str:
        """Get the schema collection name."""
        return self._schema_collection

    def register_table(self, schema: TableSchema) -> None:
        """
        Register a table schema for discovery.

        Creates/updates the schema record in the __schema collection.

        Args:
            schema: Table schema to register
        """
        # Generate searchable text and embedding
        search_text = schema.to_search_text()
        embeddings = self._embedding.embed([search_text])

        if not embeddings or not embeddings[0]:
            raise ValueError(f"Failed to generate embedding for table {schema.table_name}")

        # Upsert to schema collection
        point = {
            "id": schema.table_name,
            "vector": embeddings[0],
            "payload": schema.to_payload(),
        }

        self._vector_db.upsert(self._schema_collection, [point])

        logger.info(
            f"Registered table schema: {schema.table_name} "
            f"({len(schema.columns)} columns, {schema.row_count} rows)"
        )

    def unregister_table(self, table_name: str) -> bool:
        """
        Remove a table schema.

        Args:
            table_name: Name of table to remove

        Returns:
            True if removed, False if not found
        """
        # Use delete by filter or retrieve + delete
        # For now, we'll just try to overwrite with empty (most VDBs don't have single delete)
        # TODO: Implement proper delete when VDB supports it
        logger.info(f"Unregistered table schema: {table_name}")
        return True

    def get_table(self, table_name: str) -> TableSchema | None:
        """
        Get a specific table schema by name.

        Args:
            table_name: Name of the table

        Returns:
            TableSchema if found, None otherwise
        """
        try:
            results = self._vector_db.retrieve(
                self._schema_collection,
                ids=[table_name],
                with_payload=True,
            )

            if results:
                payload = results[0].get("payload", results[0])
                return TableSchema.from_payload(payload)

            return None

        except Exception as e:
            logger.warning(f"Failed to get table schema {table_name}: {e}")
            return None

    def search_tables(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> list[SchemaSearchResult]:
        """
        Search for tables relevant to a query.

        Uses semantic search on table/column names to find relevant tables.

        Args:
            query: User query to match against table schemas
            limit: Maximum number of results
            min_score: Minimum similarity score threshold

        Returns:
            List of matching schemas with scores
        """
        # Embed the query
        embeddings = self._embedding.embed([query])

        if not embeddings or not embeddings[0]:
            logger.warning("Failed to generate query embedding for schema search")
            return []

        # Search schema collection
        try:
            results = self._vector_db.search(
                collection_name=self._schema_collection,
                query_vector=embeddings[0],
                limit=limit,
                with_payload=True,
            )
        except Exception as e:
            logger.warning(f"Schema search failed: {e}")
            return []

        # Convert to SchemaSearchResult
        schema_results = []
        for result in results:
            score = getattr(result, "score", None) or result.get("score", 0.0)

            if score < min_score:
                continue

            payload = getattr(result, "payload", None) or result.get("payload", {})
            try:
                schema = TableSchema.from_payload(payload)
                schema_results.append(SchemaSearchResult(schema=schema, score=score))
            except Exception as e:
                logger.warning(f"Failed to parse schema from search result: {e}")
                continue

        return schema_results

    def list_tables(self) -> list[str]:
        """
        List all registered table names.

        Returns:
            List of table names
        """
        # Use scroll/list if available, otherwise search with high limit
        try:
            # Try to get all schemas with a broad search
            # This is a fallback - ideally VDB would have a list operation
            results = self._vector_db.search(
                collection_name=self._schema_collection,
                query_vector=[0.0] * 1536,  # Dummy vector
                limit=1000,
                with_payload=True,
            )

            table_names = []
            for result in results:
                payload = getattr(result, "payload", None) or result.get("payload", {})
                if "table_name" in payload:
                    table_names.append(payload["table_name"])

            return sorted(set(table_names))

        except Exception as e:
            logger.warning(f"Failed to list tables: {e}")
            return []

    def get_schema(self, table_name: str) -> TableSchema | None:
        """
        Get a specific table schema by name.

        Alias for get_table() for consistency.

        Args:
            table_name: Name of the table

        Returns:
            TableSchema if found, None otherwise
        """
        return self.get_table(table_name)

    def get_all_schemas(self) -> list[TableSchema]:
        """
        Get all registered table schemas.

        Returns:
            List of TableSchema objects
        """
        try:
            results = self._vector_db.search(
                collection_name=self._schema_collection,
                query_vector=[0.0] * 1536,  # Dummy vector
                limit=1000,
                with_payload=True,
            )

            schemas = []
            for result in results:
                payload = getattr(result, "payload", None) or result.get("payload", {})
                if "table_name" in payload:
                    try:
                        schema = TableSchema.from_payload(payload)
                        schemas.append(schema)
                    except Exception as e:
                        logger.warning(f"Failed to parse schema: {e}")
                        continue

            return schemas

        except Exception as e:
            logger.warning(f"Failed to get all schemas: {e}")
            return []

    def delete_schema(self, table_name: str) -> bool:
        """
        Delete a table schema.

        Args:
            table_name: Name of the table to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            # Check if schema exists
            existing = self.get_table(table_name)
            if existing is None:
                return False

            # Use delete method if available
            if hasattr(self._vector_db, "delete"):
                self._vector_db.delete(
                    self._schema_collection,
                    {"points": [table_name]},
                )
                logger.info(f"Deleted table schema: {table_name}")
                return True

            logger.warning(f"Vector DB doesn't support delete operation")
            return False

        except Exception as e:
            logger.warning(f"Failed to delete schema {table_name}: {e}")
            return False

    def clear(self) -> None:
        """Clear all schemas (delete the schema collection)."""
        try:
            self._vector_db.delete_collection(self._schema_collection)
            logger.info(f"Cleared schema collection: {self._schema_collection}")
        except Exception as e:
            logger.warning(f"Failed to clear schema collection: {e}")


__all__ = [
    "ColumnSchema",
    "TableSchema",
    "SchemaSearchResult",
    "SchemaStore",
    "SCHEMA_COLLECTION_SUFFIX",
]
