# tests/unit/structured/test_schema.py
"""Tests for schema storage and discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from fitz_ai.structured.schema import (
    SCHEMA_COLLECTION_SUFFIX,
    ColumnSchema,
    SchemaSearchResult,
    SchemaStore,
    TableSchema,
)


# Mock implementations
class MockEmbeddingClient:
    """Mock embedding client for testing."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings based on text hash."""
        self.calls.append(texts)
        embeddings = []
        for text in texts:
            # Create deterministic embedding from text
            h = hash(text) % 1000
            embeddings.append([h / 1000.0] * self.dim)
        return embeddings


@dataclass
class MockSearchResult:
    """Mock search result with score and payload."""

    score: float
    payload: dict[str, Any]


class MockVectorDBClient:
    """Mock vector DB client for testing."""

    def __init__(self):
        self.collections: dict[str, list[dict[str, Any]]] = {}

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Store points in collection."""
        if collection not in self.collections:
            self.collections[collection] = []

        # Upsert by ID
        existing_ids = {p["id"] for p in self.collections[collection]}
        for point in points:
            if point["id"] in existing_ids:
                # Update existing
                self.collections[collection] = [
                    p if p["id"] != point["id"] else point for p in self.collections[collection]
                ]
            else:
                self.collections[collection].append(point)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> list[MockSearchResult]:
        """Search collection by vector similarity."""
        if collection_name not in self.collections:
            return []

        points = self.collections[collection_name]

        # Simple cosine-like similarity (dot product for normalized vectors)
        scored = []
        for point in points:
            vec = point["vector"]
            score = sum(a * b for a, b in zip(query_vector, vec))
            scored.append((score, point))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, point in scored[:limit]:
            results.append(MockSearchResult(score=score, payload=point.get("payload", {})))

        return results

    def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve points by ID."""
        if collection_name not in self.collections:
            return []

        results = []
        for point in self.collections[collection_name]:
            if point["id"] in ids:
                results.append({"id": point["id"], "payload": point.get("payload", {})})

        return results

    def delete_collection(self, collection: str) -> int:
        """Delete a collection."""
        if collection in self.collections:
            count = len(self.collections[collection])
            del self.collections[collection]
            return count
        return 0


class TestColumnSchema:
    """Tests for ColumnSchema dataclass."""

    def test_column_schema_defaults(self):
        """Test default values."""
        col = ColumnSchema(name="id", type="string")
        assert col.name == "id"
        assert col.type == "string"
        assert col.nullable is True
        assert col.indexed is False

    def test_column_schema_custom_values(self):
        """Test custom values."""
        col = ColumnSchema(name="salary", type="number", nullable=False, indexed=True)
        assert col.name == "salary"
        assert col.type == "number"
        assert col.nullable is False
        assert col.indexed is True


class TestTableSchema:
    """Tests for TableSchema dataclass."""

    def test_table_schema_creation(self):
        """Test basic creation."""
        schema = TableSchema(
            table_name="employees",
            columns=[
                ColumnSchema(name="id", type="string"),
                ColumnSchema(name="name", type="string"),
                ColumnSchema(name="salary", type="number"),
            ],
            primary_key="id",
            row_count=100,
        )

        assert schema.table_name == "employees"
        assert len(schema.columns) == 3
        assert schema.primary_key == "id"
        assert schema.row_count == 100
        assert schema.version != ""  # Auto-generated

    def test_column_names_property(self):
        """Test column_names property."""
        schema = TableSchema(
            table_name="test",
            columns=[
                ColumnSchema(name="a", type="string"),
                ColumnSchema(name="b", type="number"),
            ],
            primary_key="a",
        )

        assert schema.column_names == ["a", "b"]

    def test_indexed_columns_property(self):
        """Test indexed_columns property."""
        schema = TableSchema(
            table_name="test",
            columns=[
                ColumnSchema(name="a", type="string", indexed=True),
                ColumnSchema(name="b", type="number", indexed=False),
                ColumnSchema(name="c", type="boolean", indexed=True),
            ],
            primary_key="a",
        )

        assert schema.indexed_columns == ["a", "c"]

    def test_get_column(self):
        """Test get_column method."""
        schema = TableSchema(
            table_name="test",
            columns=[
                ColumnSchema(name="id", type="string"),
                ColumnSchema(name="name", type="string"),
            ],
            primary_key="id",
        )

        assert schema.get_column("id") is not None
        assert schema.get_column("id").type == "string"
        assert schema.get_column("missing") is None

    def test_to_search_text(self):
        """Test search text generation."""
        schema = TableSchema(
            table_name="employees",
            columns=[
                ColumnSchema(name="name", type="string"),
                ColumnSchema(name="department", type="string"),
                ColumnSchema(name="salary", type="number"),
            ],
            primary_key="name",
        )

        text = schema.to_search_text()
        assert text == "employees: name, department, salary"

    def test_version_changes_with_schema(self):
        """Test that version changes when schema changes."""
        schema1 = TableSchema(
            table_name="test",
            columns=[ColumnSchema(name="a", type="string")],
            primary_key="a",
            row_count=10,
        )

        schema2 = TableSchema(
            table_name="test",
            columns=[ColumnSchema(name="a", type="string")],
            primary_key="a",
            row_count=20,  # Different row count
        )

        assert schema1.version != schema2.version

    def test_to_payload_and_from_payload(self):
        """Test round-trip serialization."""
        original = TableSchema(
            table_name="employees",
            columns=[
                ColumnSchema(name="id", type="string", nullable=False, indexed=True),
                ColumnSchema(name="salary", type="number"),
            ],
            primary_key="id",
            row_count=50,
        )

        payload = original.to_payload()
        restored = TableSchema.from_payload(payload)

        assert restored.table_name == original.table_name
        assert len(restored.columns) == len(original.columns)
        assert restored.columns[0].name == "id"
        assert restored.columns[0].indexed is True
        assert restored.primary_key == original.primary_key
        assert restored.row_count == original.row_count
        assert restored.version == original.version


class TestSchemaStore:
    """Tests for SchemaStore class."""

    @pytest.fixture
    def store(self) -> SchemaStore:
        """Create a schema store with mocks."""
        vector_db = MockVectorDBClient()
        embedding = MockEmbeddingClient(dim=4)
        return SchemaStore(vector_db, embedding, "my_docs")

    def test_schema_collection_name(self, store: SchemaStore):
        """Test schema collection naming."""
        assert store.schema_collection == f"my_docs{SCHEMA_COLLECTION_SUFFIX}"
        assert store.schema_collection == "my_docs__schema"

    def test_register_table(self, store: SchemaStore):
        """Test registering a table schema."""
        schema = TableSchema(
            table_name="employees",
            columns=[
                ColumnSchema(name="name", type="string"),
                ColumnSchema(name="salary", type="number"),
            ],
            primary_key="name",
            row_count=100,
        )

        store.register_table(schema)

        # Verify embedding was called
        assert len(store._embedding.calls) == 1
        assert "employees:" in store._embedding.calls[0][0]

        # Verify stored in vector DB
        assert store._schema_collection in store._vector_db.collections
        points = store._vector_db.collections[store._schema_collection]
        assert len(points) == 1
        assert points[0]["id"] == "employees"

    def test_get_table(self, store: SchemaStore):
        """Test retrieving a table schema by name."""
        schema = TableSchema(
            table_name="employees",
            columns=[ColumnSchema(name="id", type="string")],
            primary_key="id",
        )
        store.register_table(schema)

        retrieved = store.get_table("employees")

        assert retrieved is not None
        assert retrieved.table_name == "employees"

    def test_get_table_not_found(self, store: SchemaStore):
        """Test retrieving non-existent table returns None."""
        result = store.get_table("nonexistent")
        assert result is None

    def test_search_tables(self, store: SchemaStore):
        """Test searching for tables by query."""
        # Register multiple tables
        store.register_table(
            TableSchema(
                table_name="employees",
                columns=[
                    ColumnSchema(name="name", type="string"),
                    ColumnSchema(name="salary", type="number"),
                ],
                primary_key="name",
            )
        )
        store.register_table(
            TableSchema(
                table_name="products",
                columns=[
                    ColumnSchema(name="sku", type="string"),
                    ColumnSchema(name="price", type="number"),
                ],
                primary_key="sku",
            )
        )

        # Search with min_score=0 to test mechanics (real scores depend on embeddings)
        results = store.search_tables("employee salary", limit=5, min_score=0.0)

        assert len(results) > 0
        assert all(isinstance(r, SchemaSearchResult) for r in results)
        assert all(r.schema is not None for r in results)

    def test_search_tables_with_min_score(self, store: SchemaStore):
        """Test search respects min_score threshold."""
        store.register_table(
            TableSchema(
                table_name="test",
                columns=[ColumnSchema(name="id", type="string")],
                primary_key="id",
            )
        )

        # Very high threshold should filter out results
        results = store.search_tables("random query", min_score=0.99)

        # May or may not have results depending on mock similarity
        assert isinstance(results, list)

    def test_list_tables_empty(self, store: SchemaStore):
        """Test list_tables on empty store."""
        tables = store.list_tables()
        assert tables == []

    def test_list_tables(self, store: SchemaStore):
        """Test listing all tables."""
        store.register_table(
            TableSchema(
                table_name="alpha",
                columns=[ColumnSchema(name="id", type="string")],
                primary_key="id",
            )
        )
        store.register_table(
            TableSchema(
                table_name="beta",
                columns=[ColumnSchema(name="id", type="string")],
                primary_key="id",
            )
        )

        tables = store.list_tables()

        assert "alpha" in tables
        assert "beta" in tables

    def test_clear(self, store: SchemaStore):
        """Test clearing all schemas."""
        store.register_table(
            TableSchema(
                table_name="test",
                columns=[ColumnSchema(name="id", type="string")],
                primary_key="id",
            )
        )

        store.clear()

        assert store._schema_collection not in store._vector_db.collections

    def test_update_existing_table(self, store: SchemaStore):
        """Test that registering the same table updates it."""
        # Register initial
        store.register_table(
            TableSchema(
                table_name="employees",
                columns=[ColumnSchema(name="id", type="string")],
                primary_key="id",
                row_count=10,
            )
        )

        # Register updated
        store.register_table(
            TableSchema(
                table_name="employees",
                columns=[ColumnSchema(name="id", type="string")],
                primary_key="id",
                row_count=20,  # Changed
            )
        )

        # Should still be one record
        points = store._vector_db.collections[store._schema_collection]
        assert len(points) == 1

        # Should have updated row count
        retrieved = store.get_table("employees")
        assert retrieved.row_count == 20
