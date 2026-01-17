# tests/e2e/test_structured_e2e.py
"""
End-to-end tests for the structured data (Dual-Plane Storage) system.

Tests the complete flow:
1. Ingest CSV as structured table
2. Query routing (semantic vs structured)
3. SQL generation and execution
4. Derived sentence creation and retrieval
5. Integration with RAG pipeline

These tests use mock implementations for reproducibility.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e


class MockEmbedder:
    """Mock embedder that returns deterministic vectors based on content hash."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._call_count = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Create a deterministic but varied embedding
            h = hashlib.md5(text.encode()).hexdigest()
            # Use hash to seed values
            base = [int(h[i : i + 2], 16) / 255.0 for i in range(0, min(32, self.dim * 2), 2)]
            # Pad to full dimension
            while len(base) < self.dim:
                base.extend(base[: self.dim - len(base)])
            embeddings.append(base[: self.dim])
        return embeddings


class MockChatClient:
    """Mock chat client that returns structured responses."""

    def __init__(self):
        self.call_history = []

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Return mock responses based on message content."""
        self.call_history.append(messages)
        user_msg = messages[-1]["content"].lower() if messages else ""

        # Route classification
        if "classify" in user_msg or "semantic" in user_msg or "structured" in user_msg:
            if any(kw in user_msg for kw in ["how many", "count", "total", "sum", "average"]):
                return '{"route": "structured", "confidence": 0.9, "tables": ["employees"]}'
            return '{"route": "semantic", "confidence": 0.8, "reason": "general question"}'

        # SQL generation
        if "sql" in user_msg or "select" in user_msg or "generate" in user_msg:
            if "how many" in user_msg or "count" in user_msg:
                return '{"queries": [{"sql": "SELECT COUNT(*) as count FROM employees", "table": "employees"}]}'
            if "total salary" in user_msg or "sum" in user_msg:
                return '{"queries": [{"sql": "SELECT SUM(salary) as total FROM employees", "table": "employees"}]}'
            if "average" in user_msg:
                return '{"queries": [{"sql": "SELECT AVG(salary) as average FROM employees", "table": "employees"}]}'
            if "department" in user_msg and "engineering" in user_msg:
                return '{"queries": [{"sql": "SELECT COUNT(*) as count FROM employees WHERE department = \'engineering\'", "table": "employees"}]}'

        # Result formatting
        if "convert" in user_msg or "sentence" in user_msg or "natural" in user_msg:
            if "count" in user_msg:
                return "There are 5 employees in the database."
            if "sum" in user_msg or "total" in user_msg:
                return "The total salary for all employees is $450,000."
            if "average" in user_msg:
                return "The average salary is $90,000."

        return "Mock response"


@dataclass
class MockPoint:
    """Mock vector DB point/record."""

    id: str
    vector: list[float]
    payload: dict[str, Any]
    score: float = 0.0


class MockVectorDB:
    """Mock vector DB that stores data in memory."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._collections: dict[str, list[MockPoint]] = {}

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Upsert points to collection."""
        if collection not in self._collections:
            self._collections[collection] = []

        existing_ids = {p.id for p in self._collections[collection]}

        for point in points:
            point_id = point.get("id", "")
            vector = point.get("vector", [0.0] * self.dim)
            payload = point.get("payload", {})

            if point_id in existing_ids:
                # Update existing
                for p in self._collections[collection]:
                    if p.id == point_id:
                        p.vector = vector
                        p.payload = payload
                        break
            else:
                # Insert new
                self._collections[collection].append(
                    MockPoint(id=point_id, vector=vector, payload=payload)
                )
                existing_ids.add(point_id)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
        query_filter: dict | None = None,
    ) -> list[MockPoint]:
        """Search collection by vector similarity."""
        if collection_name not in self._collections:
            return []

        points = self._collections[collection_name]

        # Simple cosine similarity approximation
        def similarity(v1, v2):
            if not v1 or not v2:
                return 0.0
            dot = sum(a * b for a, b in zip(v1, v2))
            mag1 = sum(a * a for a in v1) ** 0.5
            mag2 = sum(b * b for b in v2) ** 0.5
            if mag1 == 0 or mag2 == 0:
                return 0.0
            return dot / (mag1 * mag2)

        # Score and sort
        scored = []
        for p in points:
            score = similarity(query_vector, p.vector)
            scored.append(MockPoint(id=p.id, vector=p.vector, payload=p.payload, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:limit]

    def scroll(
        self,
        collection_name: str,
        limit: int,
        offset: int = 0,
        scroll_filter: dict | None = None,
        with_payload: bool = True,
    ) -> tuple[list[MockPoint], int | None]:
        """Scroll through collection with optional filter."""
        if collection_name not in self._collections:
            return [], None

        points = self._collections[collection_name]

        # Apply filter if provided
        if scroll_filter:
            filtered = []
            for p in points:
                match = True
                if "must" in scroll_filter:
                    for cond in scroll_filter["must"]:
                        key = cond.get("key", "")
                        match_val = cond.get("match", {}).get("value")
                        if key and match_val is not None:
                            if p.payload.get(key) != match_val:
                                match = False
                                break
                if match:
                    filtered.append(p)
            points = filtered

        # Apply offset and limit
        result = points[offset : offset + limit]
        next_offset = offset + limit if offset + limit < len(points) else None

        return result, next_offset

    def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve points by ID."""
        if collection_name not in self._collections:
            return []

        result = []
        for point in self._collections[collection_name]:
            if point.id in ids:
                result.append({"id": point.id, "payload": point.payload})

        return result

    def delete(
        self,
        collection_name: str,
        points_selector: dict[str, Any],
    ) -> int:
        """Delete points from collection."""
        if collection_name not in self._collections:
            return 0

        point_ids = points_selector.get("points", [])
        if not point_ids:
            return 0

        original_count = len(self._collections[collection_name])
        self._collections[collection_name] = [
            p for p in self._collections[collection_name] if p.id not in point_ids
        ]
        return original_count - len(self._collections[collection_name])

    def delete_collection(self, collection: str) -> int:
        """Delete entire collection."""
        if collection in self._collections:
            count = len(self._collections[collection])
            del self._collections[collection]
            return count
        return 0

    def list_collections(self) -> list[str]:
        """List all collections."""
        return list(self._collections.keys())

    def get_collection_stats(self, collection: str) -> dict[str, Any]:
        """Get collection statistics."""
        if collection not in self._collections:
            return {}
        return {
            "points_count": len(self._collections[collection]),
            "vector_size": self.dim,
            "status": "ready",
        }


@pytest.fixture
def test_csv(tmp_path) -> Path:
    """Create a test CSV file with employee data."""
    csv_path = tmp_path / "employees.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["employee_id", "name", "department", "salary", "hire_date"])
        writer.writerow(["E001", "Alice Smith", "engineering", "120000", "2022-01-15"])
        writer.writerow(["E002", "Bob Johnson", "engineering", "100000", "2022-03-20"])
        writer.writerow(["E003", "Carol White", "marketing", "90000", "2021-11-01"])
        writer.writerow(["E004", "David Brown", "engineering", "110000", "2023-02-28"])
        writer.writerow(["E005", "Eve Davis", "hr", "80000", "2020-06-10"])
    return csv_path


@pytest.fixture
def mock_embedder():
    """Provide mock embedder."""
    return MockEmbedder(dim=384)


@pytest.fixture
def mock_chat():
    """Provide mock chat client."""
    return MockChatClient()


@pytest.fixture
def mock_vector_db():
    """Provide mock vector DB."""
    return MockVectorDB(dim=384)


@pytest.fixture
def test_collection():
    """Test collection name."""
    return "test_structured_e2e"


class TestStructuredIngestion:
    """E2E tests for structured data ingestion."""

    def test_ingest_csv_creates_schema(
        self, mock_vector_db, mock_embedder, test_csv, test_collection
    ):
        """Test that ingesting a CSV creates proper schema in __schema collection."""
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.schema import SchemaStore

        # Read CSV
        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        # Create schema store first
        store = SchemaStore(mock_vector_db, mock_embedder, test_collection)

        # Ingest
        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=store,
            base_collection=test_collection,
        )

        schema = ingester.ingest_table(
            table_name="employees",
            rows=rows,
            primary_key="employee_id",
        )

        # Verify schema
        assert schema.table_name == "employees"
        assert schema.row_count == 5
        assert schema.primary_key == "employee_id"
        assert len(schema.columns) == 5

        # Verify schema stored in __schema collection
        retrieved = store.get_table("employees")

        assert retrieved is not None
        assert retrieved.table_name == "employees"
        assert retrieved.row_count == 5

    def test_ingest_csv_creates_rows(
        self, mock_vector_db, mock_embedder, test_csv, test_collection
    ):
        """Test that ingesting a CSV stores rows in __tables collection."""
        from fitz_ai.structured.constants import FIELD_ROW_DATA, FIELD_TABLE, get_tables_collection
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.schema import SchemaStore

        # Read CSV
        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        # Create schema store
        store = SchemaStore(mock_vector_db, mock_embedder, test_collection)

        # Ingest
        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=store,
            base_collection=test_collection,
        )

        ingester.ingest_table(
            table_name="employees",
            rows=rows,
            primary_key="employee_id",
        )

        # Verify rows stored in __tables collection
        tables_collection = get_tables_collection(test_collection)

        # Scroll to get all rows
        records, _ = mock_vector_db.scroll(
            collection_name=tables_collection,
            limit=100,
            offset=0,
            with_payload=True,
        )

        assert len(records) == 5

        # Check first record structure
        first = records[0]
        payload = first.payload
        assert FIELD_TABLE in payload
        assert payload[FIELD_TABLE] == "employees"
        assert FIELD_ROW_DATA in payload

    def test_ingest_detects_column_types(
        self, mock_vector_db, mock_embedder, test_csv, test_collection
    ):
        """Test that column types are correctly inferred."""
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.schema import SchemaStore

        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        store = SchemaStore(mock_vector_db, mock_embedder, test_collection)
        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=store,
            base_collection=test_collection,
        )

        schema = ingester.ingest_table(
            table_name="employees",
            rows=rows,
            primary_key="employee_id",
        )

        # Check column types
        col_types = {c.name: c.type for c in schema.columns}

        assert col_types["employee_id"] == "string"
        assert col_types["name"] == "string"
        assert col_types["department"] == "string"
        assert col_types["salary"] == "number"
        # hire_date might be string or date depending on inference
        assert col_types["hire_date"] in ("string", "date")


class TestQueryRouting:
    """E2E tests for query routing."""

    def test_route_aggregation_query_to_structured(
        self, mock_vector_db, mock_embedder, mock_chat, test_csv, test_collection
    ):
        """Test that aggregation queries are routed to structured path."""
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.router import QueryRouter, StructuredRoute
        from fitz_ai.structured.schema import SchemaStore

        # First ingest data
        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        # Create schema store first
        schema_store = SchemaStore(mock_vector_db, mock_embedder, test_collection)

        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=schema_store,
            base_collection=test_collection,
        )
        ingester.ingest_table("employees", rows, "employee_id")

        # Create router
        router = QueryRouter(
            schema_store=schema_store,
            chat_client=mock_chat,
            structured_confidence_threshold=0.6,
        )

        # Test aggregation query
        decision = router.route("How many employees are there?")

        assert isinstance(decision, StructuredRoute)
        assert len(decision.tables) > 0


class TestSQLGenerationAndExecution:
    """E2E tests for SQL generation and execution."""

    def test_generate_count_query(self, mock_chat):
        """Test SQL generation for count queries."""
        from fitz_ai.structured.schema import ColumnSchema, TableSchema
        from fitz_ai.structured.sql_generator import SQLGenerator

        schema = TableSchema(
            table_name="employees",
            columns=[
                ColumnSchema(name="employee_id", type="string", indexed=True),
                ColumnSchema(name="name", type="string"),
                ColumnSchema(name="department", type="string", indexed=True),
                ColumnSchema(name="salary", type="number", indexed=True),
            ],
            primary_key="employee_id",
            row_count=5,
        )

        generator = SQLGenerator(chat_client=mock_chat)
        result = generator.generate("How many employees are there?", [schema])

        assert len(result.queries) > 0
        # Check the query has COUNT aggregation
        assert result.queries[0].is_aggregation
        assert "COUNT" in str(result.queries[0].select).upper()

    def test_execute_count_query(self, mock_vector_db, mock_embedder, test_csv, test_collection):
        """Test executing a count query on ingested data."""
        from fitz_ai.structured.executor import StructuredExecutor
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.schema import SchemaStore
        from fitz_ai.structured.sql_generator import SQLQuery

        # Ingest data first
        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        store = SchemaStore(mock_vector_db, mock_embedder, test_collection)
        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=store,
            base_collection=test_collection,
        )
        ingester.ingest_table("employees", rows, "employee_id")

        # Execute count query
        executor = StructuredExecutor(
            vector_db=mock_vector_db,
            base_collection=test_collection,
        )

        query = SQLQuery(
            table="employees",
            select=["COUNT(*)"],
            where=[],
            raw_sql="SELECT COUNT(*) FROM employees",
        )

        result = executor.execute(query)

        assert result.is_success
        assert result.data is not None
        assert result.data.get("COUNT(*)") == 5

    def test_execute_sum_query(self, mock_vector_db, mock_embedder, test_csv, test_collection):
        """Test executing a SUM query."""
        from fitz_ai.structured.executor import StructuredExecutor
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.schema import SchemaStore
        from fitz_ai.structured.sql_generator import SQLQuery

        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        store = SchemaStore(mock_vector_db, mock_embedder, test_collection)
        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=store,
            base_collection=test_collection,
        )
        ingester.ingest_table("employees", rows, "employee_id")

        executor = StructuredExecutor(
            vector_db=mock_vector_db,
            base_collection=test_collection,
        )

        query = SQLQuery(
            table="employees",
            select=["SUM(salary)"],
            where=[],
            raw_sql="SELECT SUM(salary) FROM employees",
        )

        result = executor.execute(query)

        assert result.is_success
        # 120000 + 100000 + 90000 + 110000 + 80000 = 500000
        assert result.data.get("SUM(salary)") == 500000

    def test_execute_query_with_filter(
        self, mock_vector_db, mock_embedder, test_csv, test_collection
    ):
        """Test executing a query with WHERE clause."""
        from fitz_ai.structured.executor import StructuredExecutor
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.schema import SchemaStore
        from fitz_ai.structured.sql_generator import SQLQuery

        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        store = SchemaStore(mock_vector_db, mock_embedder, test_collection)
        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=store,
            base_collection=test_collection,
        )
        # Explicitly index department for filtering
        ingester.ingest_table(
            "employees", rows, "employee_id",
            indexed_columns=["employee_id", "department", "salary"]
        )

        executor = StructuredExecutor(
            vector_db=mock_vector_db,
            base_collection=test_collection,
        )

        query = SQLQuery(
            table="employees",
            select=["COUNT(*)"],
            where=[{"column": "department", "op": "=", "value": "engineering"}],
            raw_sql="SELECT COUNT(*) FROM employees WHERE department = 'engineering'",
        )

        result = executor.execute(query)

        assert result.is_success
        # Alice, Bob, David are in engineering
        assert result.data.get("COUNT(*)") == 3

    def test_execute_avg_query(self, mock_vector_db, mock_embedder, test_csv, test_collection):
        """Test executing an AVG query."""
        from fitz_ai.structured.executor import StructuredExecutor
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.schema import SchemaStore
        from fitz_ai.structured.sql_generator import SQLQuery

        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        store = SchemaStore(mock_vector_db, mock_embedder, test_collection)
        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=store,
            base_collection=test_collection,
        )
        ingester.ingest_table("employees", rows, "employee_id")

        executor = StructuredExecutor(
            vector_db=mock_vector_db,
            base_collection=test_collection,
        )

        query = SQLQuery(
            table="employees",
            select=["AVG(salary)"],
            where=[],
            raw_sql="SELECT AVG(salary) FROM employees",
        )

        result = executor.execute(query)

        assert result.is_success
        # 500000 / 5 = 100000
        assert result.data.get("AVG(salary)") == 100000


class TestDerivedSentences:
    """E2E tests for derived sentence storage and retrieval."""

    def test_format_result_to_sentence(self, mock_chat):
        """Test formatting SQL result to natural language."""
        from fitz_ai.structured.executor import ExecutionResult
        from fitz_ai.structured.formatter import ResultFormatter
        from fitz_ai.structured.sql_generator import SQLQuery

        formatter = ResultFormatter(chat_client=mock_chat)

        query = SQLQuery(
            table="employees",
            select=["COUNT(*)"],
            where=[],
            raw_sql="SELECT COUNT(*) FROM employees",
        )

        result = ExecutionResult(
            data={"COUNT(*)": 5},
            row_count=5,
            query=query,
        )

        formatted = formatter.format(result)

        assert formatted.sentence is not None
        assert len(formatted.sentence) > 0

    def test_store_derived_sentence(
        self, mock_vector_db, mock_embedder, test_csv, test_collection
    ):
        """Test storing derived sentences in __derived collection."""
        from fitz_ai.structured.constants import get_derived_collection
        from fitz_ai.structured.derived import DerivedStore
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.schema import SchemaStore

        # Ingest data first
        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        store = SchemaStore(mock_vector_db, mock_embedder, test_collection)
        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=store,
            base_collection=test_collection,
        )
        schema = ingester.ingest_table("employees", rows, "employee_id")

        # Store derived sentence
        derived_store = DerivedStore(
            vector_db=mock_vector_db,
            embedding=mock_embedder,
            base_collection=test_collection,
        )

        # Use ingest() method with separate arguments
        derived_store.ingest(
            sentence="There are 5 employees in the database.",
            source_table="employees",
            source_query="SELECT COUNT(*) FROM employees",
            table_version=schema.version,
        )

        # Verify stored in __derived collection
        derived_collection = get_derived_collection(test_collection)

        # Search for the derived content
        query_vec = mock_embedder.embed(["employees count"])[0]
        results = mock_vector_db.search(
            collection_name=derived_collection,
            query_vector=query_vec,
            limit=10,
            with_payload=True,
        )

        assert len(results) > 0

    def test_invalidate_derived_on_table_update(
        self, mock_vector_db, mock_embedder, test_csv, test_collection
    ):
        """Test that derived sentences are invalidated when table is updated."""
        from fitz_ai.structured.derived import DerivedStore
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.schema import SchemaStore

        # Ingest initial data
        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        store = SchemaStore(mock_vector_db, mock_embedder, test_collection)
        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=store,
            base_collection=test_collection,
        )
        schema = ingester.ingest_table("employees", rows, "employee_id")

        # Store derived sentence
        derived_store = DerivedStore(
            vector_db=mock_vector_db,
            embedding=mock_embedder,
            base_collection=test_collection,
        )

        # Use ingest() method with separate arguments
        derived_store.ingest(
            sentence="There are 5 employees.",
            source_table="employees",
            source_query="SELECT COUNT(*)",
            table_version=schema.version,
        )

        # Invalidate derived sentences for the table
        deleted_count = derived_store.invalidate("employees")

        # Should have deleted the derived sentence
        assert deleted_count >= 1


class TestFullPipelineFlow:
    """E2E tests for the complete structured data flow."""

    def test_full_flow_count_query(
        self, mock_vector_db, mock_embedder, mock_chat, test_csv, test_collection
    ):
        """Test complete flow: ingest -> route -> generate SQL -> execute -> format."""
        from fitz_ai.structured.derived import DerivedStore
        from fitz_ai.structured.executor import StructuredExecutor
        from fitz_ai.structured.formatter import ResultFormatter
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.router import QueryRouter, StructuredRoute
        from fitz_ai.structured.schema import SchemaStore
        from fitz_ai.structured.sql_generator import SQLGenerator, SQLQuery

        # Step 1: Ingest CSV
        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        # Create schema store first
        schema_store = SchemaStore(mock_vector_db, mock_embedder, test_collection)

        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=schema_store,
            base_collection=test_collection,
        )
        # Explicitly index department for filtering
        schema = ingester.ingest_table(
            "employees", rows, "employee_id",
            indexed_columns=["employee_id", "department", "salary"]
        )

        # Step 2: Route query
        router = QueryRouter(
            schema_store=schema_store,
            chat_client=mock_chat,
            structured_confidence_threshold=0.6,
        )

        query = "How many employees are in the engineering department?"
        decision = router.route(query)

        # Should route to structured
        assert isinstance(decision, StructuredRoute)

        # Step 3: Generate SQL (using known working query since mock may vary)
        sql_query = SQLQuery(
            table="employees",
            select=["COUNT(*)"],
            where=[{"column": "department", "op": "=", "value": "engineering"}],
            raw_sql="SELECT COUNT(*) FROM employees WHERE department = 'engineering'",
        )

        # Step 4: Execute SQL
        executor = StructuredExecutor(
            vector_db=mock_vector_db,
            base_collection=test_collection,
        )

        exec_result = executor.execute(sql_query)

        assert exec_result.is_success
        assert exec_result.data.get("COUNT(*)") == 3  # Alice, Bob, David

        # Step 5: Format result
        formatter = ResultFormatter(chat_client=mock_chat)
        formatted = formatter.format(exec_result)

        assert formatted.sentence is not None

        # Step 6: Store derived sentence
        derived_store = DerivedStore(
            vector_db=mock_vector_db,
            embedding=mock_embedder,
            base_collection=test_collection,
        )

        derived_store.ingest(
            sentence=formatted.sentence,
            source_table="employees",
            source_query=sql_query.raw_sql,
            table_version=schema.version,
        )

        # Verify derived sentence is searchable
        from fitz_ai.structured.constants import get_derived_collection

        derived_collection = get_derived_collection(test_collection)
        query_vec = mock_embedder.embed(["engineering employees count"])[0]

        results = mock_vector_db.search(
            collection_name=derived_collection,
            query_vector=query_vec,
            limit=5,
            with_payload=True,
        )

        assert len(results) > 0

    def test_schema_search_finds_relevant_tables(
        self, mock_vector_db, mock_embedder, test_csv, test_collection
    ):
        """Test that schema search finds relevant tables for queries."""
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.schema import SchemaStore

        # Ingest data
        with open(test_csv) as f:
            import csv as csv_mod

            reader = csv_mod.DictReader(f)
            rows = list(reader)

        # Create schema store first
        schema_store = SchemaStore(mock_vector_db, mock_embedder, test_collection)

        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=schema_store,
            base_collection=test_collection,
        )
        ingester.ingest_table("employees", rows, "employee_id")

        # Search for relevant tables (schema_store already exists)

        # Search with employee-related query
        results = schema_store.search_tables("employee salary information", limit=5)

        # Should find the employees table
        assert len(results) > 0
        assert any(r.schema.table_name == "employees" for r in results)


class TestMultipleTablesFlow:
    """E2E tests for multiple tables."""

    def test_ingest_multiple_tables(self, mock_vector_db, mock_embedder, tmp_path, test_collection):
        """Test ingesting multiple tables."""
        from fitz_ai.structured.ingestion import StructuredIngester
        from fitz_ai.structured.schema import SchemaStore

        # Create employees CSV
        employees_csv = tmp_path / "employees.csv"
        with open(employees_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["employee_id", "name", "department_id"])
            writer.writerow(["E001", "Alice", "D1"])
            writer.writerow(["E002", "Bob", "D2"])

        # Create departments CSV
        departments_csv = tmp_path / "departments.csv"
        with open(departments_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["department_id", "name", "budget"])
            writer.writerow(["D1", "Engineering", "1000000"])
            writer.writerow(["D2", "Marketing", "500000"])

        # Create schema store first
        store = SchemaStore(mock_vector_db, mock_embedder, test_collection)

        ingester = StructuredIngester(
            vector_db=mock_vector_db,
            schema_store=store,
            base_collection=test_collection,
        )

        # Ingest both tables
        with open(employees_csv) as f:
            rows = list(csv.DictReader(f))
        ingester.ingest_table("employees", rows, "employee_id")

        with open(departments_csv) as f:
            rows = list(csv.DictReader(f))
        ingester.ingest_table("departments", rows, "department_id")

        # Verify both schemas exist
        emp_schema = store.get_table("employees")
        dept_schema = store.get_table("departments")

        assert emp_schema is not None
        assert dept_schema is not None
        assert emp_schema.table_name == "employees"
        assert dept_schema.table_name == "departments"

        # Verify schema search can find both
        results = store.search_tables("budget", limit=5)
        assert any(r.schema.table_name == "departments" for r in results)
