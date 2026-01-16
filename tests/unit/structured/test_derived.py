# tests/unit/structured/test_derived.py
"""Tests for derived sentence storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from fitz_ai.structured.derived import (
    FIELD_CONTENT,
    FIELD_DERIVED,
    FIELD_GENERATED_AT,
    FIELD_SOURCE_QUERY,
    FIELD_SOURCE_TABLE,
    FIELD_TABLE_VERSION,
    DerivedRecord,
    DerivedStore,
)


@dataclass
class MockScrollRecord:
    """Mock scroll record."""

    id: str
    payload: dict[str, Any]


class MockEmbeddingClient:
    """Mock embedding client."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1 * i] * self.dim for i in range(len(texts))]


class MockVectorDBClient:
    """Mock vector DB client for derived storage."""

    def __init__(self):
        self.collections: dict[str, list[dict[str, Any]]] = {}
        self.upsert_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    def upsert(self, collection_name: str, points: list[dict[str, Any]]) -> None:
        self.upsert_calls.append({
            "collection": collection_name,
            "points": points,
        })
        if collection_name not in self.collections:
            self.collections[collection_name] = []

        existing_ids = {p["id"] for p in self.collections[collection_name]}
        for point in points:
            if point["id"] in existing_ids:
                self.collections[collection_name] = [
                    p if p["id"] != point["id"] else point
                    for p in self.collections[collection_name]
                ]
            else:
                self.collections[collection_name].append(point)

    def scroll(
        self,
        collection_name: str,
        limit: int,
        offset: int = 0,
        scroll_filter: dict[str, Any] | None = None,
        with_payload: bool = True,
    ) -> tuple[list[MockScrollRecord], int | None]:
        if collection_name not in self.collections:
            return [], None

        points = self.collections[collection_name]

        # Simple filter matching
        if scroll_filter:
            filtered = []
            for p in points:
                payload = p.get("payload", {})
                match = True

                # Handle must conditions
                if "must" in scroll_filter:
                    for cond in scroll_filter["must"]:
                        key = cond.get("key")
                        expected = cond.get("match", {}).get("value")
                        if payload.get(key) != expected:
                            match = False
                            break

                # Handle must_not conditions
                if match and "must_not" in scroll_filter:
                    for cond in scroll_filter["must_not"]:
                        key = cond.get("key")
                        excluded = cond.get("match", {}).get("value")
                        if payload.get(key) == excluded:
                            match = False
                            break

                if match:
                    filtered.append(p)
            points = filtered

        # Paginate
        start = offset
        end = offset + limit
        batch = points[start:end]

        records = [
            MockScrollRecord(id=p["id"], payload=p.get("payload", {}))
            for p in batch
        ]
        next_offset = end if end < len(points) else None

        return records, next_offset

    def delete(
        self,
        collection_name: str,
        points_selector: dict[str, Any],
    ) -> int:
        self.delete_calls.append({
            "collection": collection_name,
            "selector": points_selector,
        })

        if collection_name not in self.collections:
            return 0

        ids_to_delete = set(points_selector.get("points", []))
        original_count = len(self.collections[collection_name])
        self.collections[collection_name] = [
            p for p in self.collections[collection_name]
            if p["id"] not in ids_to_delete
        ]
        return original_count - len(self.collections[collection_name])


@pytest.fixture
def vector_db() -> MockVectorDBClient:
    return MockVectorDBClient()


@pytest.fixture
def embedding() -> MockEmbeddingClient:
    return MockEmbeddingClient(dim=4)


@pytest.fixture
def store(vector_db: MockVectorDBClient, embedding: MockEmbeddingClient) -> DerivedStore:
    return DerivedStore(vector_db, embedding, "test")


class TestDerivedRecord:
    """Tests for DerivedRecord dataclass."""

    def test_to_payload(self):
        """Test converting record to payload."""
        now = datetime.now(timezone.utc)
        record = DerivedRecord(
            id="abc123",
            content="There are 42 employees.",
            source_table="employees",
            source_query="SELECT COUNT(*) FROM employees",
            table_version="v1",
            generated_at=now,
        )

        payload = record.to_payload()

        assert payload[FIELD_DERIVED] is True
        assert payload[FIELD_SOURCE_TABLE] == "employees"
        assert payload[FIELD_SOURCE_QUERY] == "SELECT COUNT(*) FROM employees"
        assert payload[FIELD_TABLE_VERSION] == "v1"
        assert payload[FIELD_CONTENT] == "There are 42 employees."
        assert FIELD_GENERATED_AT in payload

    def test_from_payload(self):
        """Test creating record from payload."""
        payload = {
            FIELD_DERIVED: True,
            FIELD_SOURCE_TABLE: "sales",
            FIELD_SOURCE_QUERY: "SELECT SUM(revenue) FROM sales",
            FIELD_TABLE_VERSION: "v2",
            FIELD_GENERATED_AT: "2024-01-15T10:30:00+00:00",
            FIELD_CONTENT: "Total revenue is $1M.",
        }

        record = DerivedRecord.from_payload("xyz789", payload)

        assert record.id == "xyz789"
        assert record.content == "Total revenue is $1M."
        assert record.source_table == "sales"
        assert record.source_query == "SELECT SUM(revenue) FROM sales"
        assert record.table_version == "v2"

    def test_from_payload_missing_fields(self):
        """Test from_payload with missing fields."""
        payload = {FIELD_CONTENT: "Some content"}

        record = DerivedRecord.from_payload("id1", payload)

        assert record.id == "id1"
        assert record.content == "Some content"
        assert record.source_table == ""
        assert record.source_query == ""


class TestDerivedStore:
    """Tests for DerivedStore class."""

    def test_collection_name(self, store: DerivedStore):
        """Test derived collection name."""
        assert store.collection_name == "test__derived"

    def test_ingest_single(
        self, store: DerivedStore, vector_db: MockVectorDBClient, embedding: MockEmbeddingClient
    ):
        """Test ingesting a single derived sentence."""
        record = store.ingest(
            sentence="There are 42 employees.",
            source_table="employees",
            source_query="SELECT COUNT(*) FROM employees",
            table_version="v1",
        )

        assert isinstance(record, DerivedRecord)
        assert record.content == "There are 42 employees."
        assert record.source_table == "employees"
        assert len(vector_db.upsert_calls) == 1
        assert len(embedding.calls) == 1
        assert embedding.calls[0] == ["There are 42 employees."]

    def test_ingest_creates_embedding(
        self, store: DerivedStore, vector_db: MockVectorDBClient
    ):
        """Test that ingest creates embedding for sentence."""
        store.ingest(
            sentence="Test sentence.",
            source_table="test",
            source_query="SELECT 1",
            table_version="v1",
        )

        point = vector_db.upsert_calls[0]["points"][0]
        assert "vector" in point
        assert len(point["vector"]) == 4  # dim from mock

    def test_ingest_stores_payload(self, store: DerivedStore, vector_db: MockVectorDBClient):
        """Test that ingest stores correct payload."""
        store.ingest(
            sentence="Revenue is high.",
            source_table="sales",
            source_query="SELECT SUM(revenue) FROM sales",
            table_version="abc123",
        )

        point = vector_db.upsert_calls[0]["points"][0]
        payload = point["payload"]

        assert payload[FIELD_DERIVED] is True
        assert payload[FIELD_SOURCE_TABLE] == "sales"
        assert payload[FIELD_SOURCE_QUERY] == "SELECT SUM(revenue) FROM sales"
        assert payload[FIELD_TABLE_VERSION] == "abc123"
        assert payload[FIELD_CONTENT] == "Revenue is high."

    def test_ingest_batch(
        self, store: DerivedStore, vector_db: MockVectorDBClient, embedding: MockEmbeddingClient
    ):
        """Test ingesting multiple sentences."""
        records = store.ingest_batch(
            sentences=["Sentence 1.", "Sentence 2.", "Sentence 3."],
            source_table="data",
            source_queries=["Q1", "Q2", "Q3"],
            table_version="v1",
        )

        assert len(records) == 3
        assert len(embedding.calls) == 1
        assert len(embedding.calls[0]) == 3  # All sentences embedded together

    def test_ingest_batch_empty(self, store: DerivedStore):
        """Test ingesting empty list."""
        records = store.ingest_batch(
            sentences=[],
            source_table="data",
            source_queries=[],
            table_version="v1",
        )

        assert records == []

    def test_invalidate_table(
        self, store: DerivedStore, vector_db: MockVectorDBClient
    ):
        """Test invalidating all derived sentences for a table."""
        # Ingest some sentences
        store.ingest("S1", "table1", "Q1", "v1")
        store.ingest("S2", "table1", "Q2", "v1")
        store.ingest("S3", "table2", "Q3", "v1")

        # Invalidate table1
        deleted = store.invalidate("table1")

        assert deleted == 2

    def test_invalidate_empty_table(self, store: DerivedStore):
        """Test invalidating when no sentences exist."""
        deleted = store.invalidate("nonexistent")
        assert deleted == 0

    def test_invalidate_stale(
        self, store: DerivedStore, vector_db: MockVectorDBClient
    ):
        """Test invalidating stale sentences by version."""
        # Ingest with old version
        store.ingest("Old sentence.", "data", "Q1", "v1")
        # Ingest with current version
        store.ingest("New sentence.", "data", "Q2", "v2")

        # Invalidate stale (not v2)
        deleted = store.invalidate_stale("data", "v2")

        assert deleted == 1

    def test_get_by_table(
        self, store: DerivedStore, vector_db: MockVectorDBClient
    ):
        """Test retrieving all derived sentences for a table."""
        store.ingest("S1 for T1.", "table1", "Q1", "v1")
        store.ingest("S2 for T1.", "table1", "Q2", "v1")
        store.ingest("S3 for T2.", "table2", "Q3", "v1")

        records = store.get_by_table("table1")

        assert len(records) == 2
        assert all(r.source_table == "table1" for r in records)

    def test_get_by_table_empty(self, store: DerivedStore):
        """Test get_by_table when no sentences exist."""
        records = store.get_by_table("nonexistent")
        assert records == []

    def test_deterministic_id(self, store: DerivedStore):
        """Test that same sentence + table produces same ID."""
        r1 = store.ingest("Same sentence.", "same_table", "Q1", "v1")
        r2 = store.ingest("Same sentence.", "same_table", "Q2", "v2")

        assert r1.id == r2.id  # Same ID for same content

    def test_different_tables_different_ids(self, store: DerivedStore):
        """Test that same sentence in different tables has different IDs."""
        r1 = store.ingest("Same sentence.", "table1", "Q1", "v1")
        r2 = store.ingest("Same sentence.", "table2", "Q2", "v1")

        assert r1.id != r2.id  # Different IDs for different tables


class TestDerivedStoreFields:
    """Tests for field constants."""

    def test_field_constants_exist(self):
        """Test that all field constants are defined."""
        assert FIELD_DERIVED == "__derived"
        assert FIELD_SOURCE_TABLE == "__source_table"
        assert FIELD_SOURCE_QUERY == "__source_query"
        assert FIELD_TABLE_VERSION == "__table_version"
        assert FIELD_GENERATED_AT == "__generated_at"
        assert FIELD_CONTENT == "content"
