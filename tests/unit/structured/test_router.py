# tests/unit/structured/test_router.py
"""Tests for semantic query routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from fitz_ai.structured.router import (
    QueryRouter,
    SemanticRoute,
    StructuredRoute,
    _format_schemas_for_prompt,
    _parse_classification_response,
)
from fitz_ai.structured.schema import ColumnSchema, SchemaStore, TableSchema


# Mock implementations
@dataclass
class MockSearchResult:
    """Mock search result."""

    score: float
    payload: dict[str, Any]


class MockEmbeddingClient:
    """Mock embedding client."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """Generate embedding that creates predictable similarity."""
        text_lower = text.lower()
        vec = [0.1] * self.dim

        if "employee" in text_lower or "salary" in text_lower:
            vec[0] = 0.9
        if "product" in text_lower or "price" in text_lower:
            vec[1] = 0.9
        if "order" in text_lower or "revenue" in text_lower:
            vec[2] = 0.9

        return vec


class MockVectorDBClient:
    """Mock vector DB client."""

    def __init__(self):
        self.collections: dict[str, list[dict[str, Any]]] = {}

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        if collection not in self.collections:
            self.collections[collection] = []

        existing_ids = {p["id"] for p in self.collections[collection]}
        for point in points:
            if point["id"] in existing_ids:
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
        if collection_name not in self.collections:
            return []

        points = self.collections[collection_name]
        scored = []
        for point in points:
            vec = point["vector"]
            score = sum(a * b for a, b in zip(query_vector, vec))
            scored.append((score, point))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [MockSearchResult(score=s, payload=p.get("payload", {})) for s, p in scored[:limit]]

    def retrieve(self, collection_name: str, ids: list[str], with_payload: bool = True):
        if collection_name not in self.collections:
            return []
        return [
            {"id": p["id"], "payload": p.get("payload", {})}
            for p in self.collections[collection_name]
            if p["id"] in ids
        ]

    def delete_collection(self, collection: str) -> int:
        if collection in self.collections:
            count = len(self.collections[collection])
            del self.collections[collection]
            return count
        return 0


class MockChatClient:
    """Mock chat client for semantic classification."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None):
        """
        Initialize with optional predefined responses.

        Args:
            responses: Dict mapping query substrings to classification responses
        """
        self.responses = responses or {}
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """Return mock classification response."""
        self.calls.append(messages)

        # Extract query from the prompt
        prompt = messages[0]["content"] if messages else ""

        # Check predefined responses
        for query_substring, response in self.responses.items():
            if query_substring.lower() in prompt.lower():
                return json.dumps(response)

        # Default: classify based on common patterns
        prompt_lower = prompt.lower()

        # Structured patterns
        structured_patterns = [
            "how many",
            "count",
            "total",
            "sum",
            "average",
            "list all",
            "show all",
            "top",
            "highest",
            "lowest",
            "maximum",
            "minimum",
            "above",
            "below",
            "more than",
            "less than",
            "between",
            "headcount",
            "tally",
            "enumerate",
            "breakdown",
        ]

        for pattern in structured_patterns:
            if pattern in prompt_lower:
                return json.dumps(
                    {
                        "route": "structured",
                        "confidence": 0.85,
                        "query_type": "aggregation",
                        "reason": f"Query contains '{pattern}' pattern",
                    }
                )

        # Default to semantic
        return json.dumps(
            {
                "route": "semantic",
                "confidence": 0.9,
                "query_type": "",
                "reason": "Query asks for explanation or concept",
            }
        )


@pytest.fixture
def employees_schema() -> TableSchema:
    """Create employees table schema."""
    return TableSchema(
        table_name="employees",
        columns=[
            ColumnSchema(name="id", type="string"),
            ColumnSchema(name="name", type="string"),
            ColumnSchema(name="department", type="string", indexed=True),
            ColumnSchema(name="salary", type="number", indexed=True),
            ColumnSchema(name="hire_date", type="date"),
        ],
        primary_key="id",
        row_count=100,
    )


@pytest.fixture
def products_schema() -> TableSchema:
    """Create products table schema."""
    return TableSchema(
        table_name="products",
        columns=[
            ColumnSchema(name="sku", type="string"),
            ColumnSchema(name="name", type="string"),
            ColumnSchema(name="price", type="number", indexed=True),
            ColumnSchema(name="category", type="string", indexed=True),
        ],
        primary_key="sku",
        row_count=50,
    )


@pytest.fixture
def schema_store(employees_schema: TableSchema, products_schema: TableSchema) -> SchemaStore:
    """Create schema store with test tables."""
    vector_db = MockVectorDBClient()
    embedding = MockEmbeddingClient(dim=4)
    store = SchemaStore(vector_db, embedding, "test")
    store.register_table(employees_schema)
    store.register_table(products_schema)
    return store


def create_mock_chat_factory(responses: dict[str, dict] | None = None):
    """Create a mock chat factory that returns a mock client."""
    client = MockChatClient(responses)

    def factory(tier: str = "fast") -> MockChatClient:
        return client

    return factory, client


@pytest.fixture
def chat_factory():
    """Create mock chat factory."""
    factory, client = create_mock_chat_factory()
    return factory


@pytest.fixture
def router(schema_store: SchemaStore, chat_factory) -> QueryRouter:
    """Create query router with mocks."""
    return QueryRouter(
        schema_store=schema_store,
        chat_factory=chat_factory,
        schema_match_threshold=0.3,
        structured_confidence_threshold=0.6,
    )


class TestParseClassificationResponse:
    """Tests for response parsing."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"route": "structured", "confidence": 0.9, "query_type": "aggregation"}'
        result = _parse_classification_response(response)

        assert result["route"] == "structured"
        assert result["confidence"] == 0.9

    def test_parse_json_with_code_block(self):
        """Test parsing JSON wrapped in code block."""
        response = '```json\n{"route": "semantic", "confidence": 0.8}\n```'
        result = _parse_classification_response(response)

        assert result["route"] == "semantic"

    def test_parse_invalid_json_defaults_to_semantic(self):
        """Test that invalid JSON defaults to semantic."""
        response = "This is not JSON at all"
        result = _parse_classification_response(response)

        assert result["route"] == "semantic"
        assert result["confidence"] == 0.0


class TestFormatSchemasForPrompt:
    """Tests for schema formatting."""

    def test_format_single_schema(self, employees_schema: TableSchema):
        """Test formatting single schema."""
        result = _format_schemas_for_prompt([employees_schema])

        assert "employees" in result
        assert "salary (number)" in result
        assert "100 rows" in result

    def test_format_multiple_schemas(
        self, employees_schema: TableSchema, products_schema: TableSchema
    ):
        """Test formatting multiple schemas."""
        result = _format_schemas_for_prompt([employees_schema, products_schema])

        assert "employees" in result
        assert "products" in result

    def test_format_empty_list(self):
        """Test formatting empty list."""
        result = _format_schemas_for_prompt([])
        assert "No tables available" in result


class TestQueryRouter:
    """Tests for QueryRouter semantic classification."""

    def test_route_aggregation_query(self, router: QueryRouter):
        """Test routing aggregation queries to structured."""
        route = router.route("How many employees are there?")

        assert isinstance(route, StructuredRoute)
        assert route.confidence >= 0.6

    def test_route_semantic_query(self, router: QueryRouter):
        """Test routing semantic queries."""
        route = router.route("Explain the concept of machine learning")

        assert isinstance(route, SemanticRoute)

    def test_route_no_schema_match(self, schema_store: SchemaStore, chat_factory):
        """Test routing when no schema matches."""
        router = QueryRouter(
            schema_store=schema_store,
            chat_factory=chat_factory,
            schema_match_threshold=0.99,  # Very high threshold
        )

        route = router.route("random query about nothing")

        assert isinstance(route, SemanticRoute)
        assert "No matching table" in route.reason

    def test_route_total_query(self, router: QueryRouter):
        """Test routing total/sum queries."""
        route = router.route("What is the total salary expense?")

        assert isinstance(route, StructuredRoute)

    def test_route_list_query(self, router: QueryRouter):
        """Test routing list queries."""
        route = router.route("List all employees in engineering")

        assert isinstance(route, StructuredRoute)

    def test_route_average_query(self, router: QueryRouter):
        """Test routing average queries."""
        route = router.route("What is the average product price?")

        assert isinstance(route, StructuredRoute)

    def test_route_top_n_query(self, router: QueryRouter):
        """Test routing top N queries."""
        route = router.route("Top 5 highest paid employees")

        assert isinstance(route, StructuredRoute)

    def test_route_semantic_headcount(self, router: QueryRouter):
        """Test that semantic variations like 'headcount' work."""
        # Include 'employee' so schema store can match it
        route = router.route("What's the employee headcount in engineering?")

        assert isinstance(route, StructuredRoute)

    def test_route_tally(self, router: QueryRouter):
        """Test that 'tally' routes to structured."""
        route = router.route("Give me a tally of products by category")

        assert isinstance(route, StructuredRoute)

    def test_primary_table_property(self, router: QueryRouter):
        """Test primary_table property."""
        route = router.route("How many employees?")

        assert isinstance(route, StructuredRoute)
        assert route.primary_table.table_name == "employees"

    def test_should_use_structured_convenience(self, router: QueryRouter):
        """Test should_use_structured convenience method."""
        assert router.should_use_structured("How many employees?") is True
        assert router.should_use_structured("What is deep learning?") is False

    def test_chat_client_called(self, schema_store: SchemaStore):
        """Test that chat client is called for classification."""
        factory, client = create_mock_chat_factory()
        router = QueryRouter(
            schema_store=schema_store,
            chat_factory=factory,
        )

        router.route("How many employees?")

        assert len(client.calls) == 1
        assert "employees" in client.calls[0][0]["content"].lower()

    def test_low_confidence_routes_to_semantic(self, schema_store: SchemaStore):
        """Test that low confidence classification routes to semantic."""
        # Create chat factory that returns low confidence
        low_conf_factory, _ = create_mock_chat_factory(
            responses={
                "employees": {
                    "route": "structured",
                    "confidence": 0.4,  # Below threshold
                    "query_type": "aggregation",
                    "reason": "Uncertain classification",
                }
            }
        )

        router = QueryRouter(
            schema_store=schema_store,
            chat_factory=low_conf_factory,
            structured_confidence_threshold=0.6,
        )

        route = router.route("Something about employees")

        assert isinstance(route, SemanticRoute)


class TestRouteDecisionTypes:
    """Tests for route decision types."""

    def test_semantic_route_has_reason(self):
        """Test SemanticRoute has reason."""
        route = SemanticRoute(reason="No table match")
        assert route.reason == "No table match"

    def test_structured_route_has_tables(self, employees_schema: TableSchema):
        """Test StructuredRoute has tables and confidence."""
        route = StructuredRoute(
            tables=[employees_schema],
            scores=[0.85],
            query_type="aggregation",
            confidence=0.9,
        )

        assert len(route.tables) == 1
        assert route.tables[0].table_name == "employees"
        assert route.confidence == 0.9
        assert route.query_type == "aggregation"
        assert route.primary_table == employees_schema
