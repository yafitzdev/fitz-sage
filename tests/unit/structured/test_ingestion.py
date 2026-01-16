# tests/unit/structured/test_ingestion.py
"""Tests for structured data ingestion."""

from __future__ import annotations

from typing import Any

import pytest

from fitz_ai.structured.constants import (
    FIELD_PRIMARY_KEY,
    FIELD_ROW_DATA,
    FIELD_TABLE,
    MAX_SCAN_ROWS,
    get_tables_collection,
)
from fitz_ai.structured.ingestion import (
    MissingPrimaryKeyError,
    StructuredIngester,
    TableTooLargeError,
)
from fitz_ai.structured.schema import SchemaStore
from fitz_ai.structured.types import TYPE_NUMBER, TYPE_STRING


# Mock implementations
class MockEmbeddingClient:
    """Mock embedding client."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings."""
        return [[hash(t) % 1000 / 1000.0] * self.dim for t in texts]


class MockVectorDBClient:
    """Mock vector DB client."""

    def __init__(self):
        self.collections: dict[str, list[dict[str, Any]]] = {}
        self.upsert_calls: list[tuple[str, list[dict]]] = []

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Store points."""
        self.upsert_calls.append((collection, points))
        if collection not in self.collections:
            self.collections[collection] = []

        existing_ids = {p["id"] for p in self.collections[collection]}
        for point in points:
            if point["id"] in existing_ids:
                self.collections[collection] = [
                    p if p["id"] != point["id"] else point
                    for p in self.collections[collection]
                ]
            else:
                self.collections[collection].append(point)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> list[Any]:
        """Search (minimal implementation for schema store)."""
        if collection_name not in self.collections:
            return []
        return self.collections[collection_name][:limit]

    def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve by ID."""
        if collection_name not in self.collections:
            return []
        return [
            {"id": p["id"], "payload": p.get("payload", {})}
            for p in self.collections[collection_name]
            if p["id"] in ids
        ]

    def delete_collection(self, collection: str) -> int:
        """Delete collection."""
        if collection in self.collections:
            count = len(self.collections[collection])
            del self.collections[collection]
            return count
        return 0


@pytest.fixture
def mock_db() -> MockVectorDBClient:
    """Create mock vector DB."""
    return MockVectorDBClient()


@pytest.fixture
def schema_store(mock_db: MockVectorDBClient) -> SchemaStore:
    """Create schema store with mocks."""
    embedding = MockEmbeddingClient(dim=4)
    return SchemaStore(mock_db, embedding, "test_collection")


@pytest.fixture
def ingester(mock_db: MockVectorDBClient, schema_store: SchemaStore) -> StructuredIngester:
    """Create ingester with mocks."""
    return StructuredIngester(
        vector_db=mock_db,
        schema_store=schema_store,
        base_collection="test_collection",
        vector_dim=4,
    )


class TestStructuredIngester:
    """Tests for StructuredIngester class."""

    def test_tables_collection_name(self, ingester: StructuredIngester):
        """Test tables collection naming."""
        assert ingester.tables_collection == "test_collection__tables"

    def test_ingest_empty_table(self, ingester: StructuredIngester):
        """Test ingesting empty table."""
        schema = ingester.ingest_table(
            table_name="empty",
            rows=[],
            primary_key="id",
        )

        assert schema.table_name == "empty"
        assert schema.row_count == 0
        assert len(schema.columns) == 0

    def test_ingest_simple_table(
        self, ingester: StructuredIngester, mock_db: MockVectorDBClient
    ):
        """Test ingesting a simple table."""
        rows = [
            {"id": "1", "name": "Alice", "age": 30},
            {"id": "2", "name": "Bob", "age": 25},
            {"id": "3", "name": "Charlie", "age": 35},
        ]

        schema = ingester.ingest_table(
            table_name="users",
            rows=rows,
            primary_key="id",
        )

        # Verify schema
        assert schema.table_name == "users"
        assert schema.row_count == 3
        assert schema.primary_key == "id"
        assert len(schema.columns) == 3

        # Verify column types inferred
        id_col = schema.get_column("id")
        assert id_col is not None
        assert id_col.type == TYPE_STRING  # "1", "2", "3" are strings

        age_col = schema.get_column("age")
        assert age_col is not None
        assert age_col.type == TYPE_NUMBER

        # Verify rows stored in tables collection
        tables_coll = get_tables_collection("test_collection")
        assert tables_coll in mock_db.collections
        stored_points = mock_db.collections[tables_coll]
        assert len(stored_points) == 3

    def test_ingest_with_custom_indexed_columns(
        self, ingester: StructuredIngester, mock_db: MockVectorDBClient
    ):
        """Test ingesting with explicit indexed columns."""
        rows = [
            {"id": "1", "name": "Alice", "dept": "eng", "level": 3},
            {"id": "2", "name": "Bob", "dept": "sales", "level": 2},
        ]

        schema = ingester.ingest_table(
            table_name="employees",
            rows=rows,
            primary_key="id",
            indexed_columns=["id", "dept"],
        )

        # Verify indexed columns
        assert schema.get_column("id").indexed is True
        assert schema.get_column("dept").indexed is True
        assert schema.get_column("name").indexed is False
        assert schema.get_column("level").indexed is False

    def test_row_payload_structure(
        self, ingester: StructuredIngester, mock_db: MockVectorDBClient
    ):
        """Test that row payloads have correct structure."""
        rows = [
            {"id": "emp1", "name": "Alice", "salary": 100000},
        ]

        ingester.ingest_table(
            table_name="employees",
            rows=rows,
            primary_key="id",
            indexed_columns=["id", "salary"],
        )

        # Get stored point
        tables_coll = get_tables_collection("test_collection")
        point = mock_db.collections[tables_coll][0]

        # Verify structure
        assert point["id"] == "employees:emp1"
        assert point["vector"] == [0.0] * 4  # Zero vector

        payload = point["payload"]
        assert payload[FIELD_TABLE] == "employees"
        assert payload[FIELD_PRIMARY_KEY] == "emp1"
        assert payload["id"] == "emp1"  # Indexed column
        assert payload["salary"] == 100000  # Indexed column
        assert FIELD_ROW_DATA in payload
        assert payload[FIELD_ROW_DATA]["name"] == "Alice"

    def test_ingest_table_too_large(self, ingester: StructuredIngester):
        """Test that large tables raise error."""
        # Create rows exceeding limit
        rows = [{"id": str(i), "value": i} for i in range(MAX_SCAN_ROWS + 1)]

        with pytest.raises(TableTooLargeError) as exc_info:
            ingester.ingest_table(
                table_name="huge",
                rows=rows,
                primary_key="id",
            )

        assert exc_info.value.table_name == "huge"
        assert exc_info.value.row_count == MAX_SCAN_ROWS + 1

    def test_ingest_missing_primary_key(self, ingester: StructuredIngester):
        """Test that missing primary key raises error."""
        rows = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]

        with pytest.raises(MissingPrimaryKeyError) as exc_info:
            ingester.ingest_table(
                table_name="users",
                rows=rows,
                primary_key="id",  # Not in rows
            )

        assert exc_info.value.primary_key == "id"

    def test_ingest_registers_schema(
        self, ingester: StructuredIngester, schema_store: SchemaStore
    ):
        """Test that ingestion registers schema for discovery."""
        rows = [
            {"id": "1", "name": "Test"},
        ]

        ingester.ingest_table(
            table_name="test_table",
            rows=rows,
            primary_key="id",
        )

        # Should be able to retrieve schema
        retrieved = schema_store.get_table("test_table")
        assert retrieved is not None
        assert retrieved.table_name == "test_table"

    def test_ingest_type_coercion(
        self, ingester: StructuredIngester, mock_db: MockVectorDBClient
    ):
        """Test that indexed columns are type-coerced."""
        rows = [
            {"id": "1", "amount": "1000", "active": "yes"},  # String values
        ]

        ingester.ingest_table(
            table_name="orders",
            rows=rows,
            primary_key="id",
            indexed_columns=["id", "amount", "active"],
        )

        # Get stored point
        tables_coll = get_tables_collection("test_collection")
        point = mock_db.collections[tables_coll][0]
        payload = point["payload"]

        # amount should be coerced to number
        assert payload["amount"] == 1000
        # active should be coerced to boolean
        assert payload["active"] is True

    def test_ingest_batching(
        self, ingester: StructuredIngester, mock_db: MockVectorDBClient
    ):
        """Test that large tables are upserted in batches."""
        # Create 250 rows (should be 3 batches of 100)
        rows = [{"id": str(i), "value": i} for i in range(250)]

        ingester.ingest_table(
            table_name="batched",
            rows=rows,
            primary_key="id",
        )

        # Count upsert calls to tables collection
        tables_coll = get_tables_collection("test_collection")
        table_upserts = [
            call for call in mock_db.upsert_calls if call[0] == tables_coll
        ]

        # Should have multiple batches
        assert len(table_upserts) == 3  # 100 + 100 + 50

        # Total stored should be 250
        assert len(mock_db.collections[tables_coll]) == 250


class TestTableTooLargeError:
    """Tests for TableTooLargeError."""

    def test_error_message(self):
        """Test error message format."""
        error = TableTooLargeError("big_table", 15000)

        assert "big_table" in str(error)
        assert "15000" in str(error)
        assert str(MAX_SCAN_ROWS) in str(error)


class TestMissingPrimaryKeyError:
    """Tests for MissingPrimaryKeyError."""

    def test_error_message(self):
        """Test error message format."""
        error = MissingPrimaryKeyError("users", "user_id")

        assert "users" in str(error)
        assert "user_id" in str(error)
