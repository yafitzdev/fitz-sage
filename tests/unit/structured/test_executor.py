# tests/unit/structured/test_executor.py
"""Tests for SQL execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from fitz_ai.structured.constants import FIELD_ROW_DATA, FIELD_TABLE
from fitz_ai.structured.executor import (
    QueryLimitExceededError,
    StructuredExecutor,
    _aggregate_avg,
    _aggregate_count,
    _aggregate_group_concat,
    _aggregate_max,
    _aggregate_min,
    _aggregate_sum,
    _build_scroll_filter,
    _condition_to_filter,
)
from fitz_ai.structured.sql_generator import SQLQuery


@dataclass
class MockScrollRecord:
    """Mock scroll record with payload."""

    payload: dict[str, Any]


class MockVectorDBClient:
    """Mock vector DB client with scroll support."""

    def __init__(self, rows: list[dict[str, Any]] | None = None):
        self.rows = rows or []
        self.scroll_calls: list[dict[str, Any]] = []

    def scroll(
        self,
        collection_name: str,
        limit: int,
        offset: int = 0,
        scroll_filter: dict[str, Any] | None = None,
        with_payload: bool = True,
    ) -> tuple[list[MockScrollRecord], int | None]:
        """Return mock rows."""
        self.scroll_calls.append(
            {
                "collection": collection_name,
                "limit": limit,
                "offset": offset,
                "filter": scroll_filter,
            }
        )

        # Filter rows by table if filter specifies it
        filtered_rows = self.rows
        if scroll_filter:
            table_filter = None
            if "must" in scroll_filter:
                for cond in scroll_filter["must"]:
                    if cond.get("key") == FIELD_TABLE:
                        table_filter = cond.get("match", {}).get("value")
            elif scroll_filter.get("key") == FIELD_TABLE:
                table_filter = scroll_filter.get("match", {}).get("value")

            if table_filter:
                filtered_rows = [r for r in self.rows if r.get(FIELD_TABLE) == table_filter]

        # Paginate
        start = offset
        end = offset + limit
        batch = filtered_rows[start:end]

        records = [MockScrollRecord(payload=r) for r in batch]
        next_offset = end if end < len(filtered_rows) else None

        return records, next_offset


class TestConditionToFilter:
    """Tests for condition to filter conversion."""

    def test_equality(self):
        """Test equality condition."""
        result = _condition_to_filter(
            {
                "column": "department",
                "op": "=",
                "value": "engineering",
            }
        )

        assert result == {"key": "department", "match": {"value": "engineering"}}

    def test_greater_than(self):
        """Test greater than condition."""
        result = _condition_to_filter(
            {
                "column": "salary",
                "op": ">",
                "value": 100000,
            }
        )

        assert result == {"key": "salary", "range": {"gt": 100000}}

    def test_less_than_or_equal(self):
        """Test less than or equal condition."""
        result = _condition_to_filter(
            {
                "column": "level",
                "op": "<=",
                "value": 5,
            }
        )

        assert result == {"key": "level", "range": {"lte": 5}}

    def test_between(self):
        """Test BETWEEN condition."""
        result = _condition_to_filter(
            {
                "column": "salary",
                "op": "BETWEEN",
                "value": [50000, 100000],
            }
        )

        assert result == {"key": "salary", "range": {"gte": 50000, "lte": 100000}}

    def test_in(self):
        """Test IN condition."""
        result = _condition_to_filter(
            {
                "column": "department",
                "op": "IN",
                "value": ["eng", "sales"],
            }
        )

        assert "should" in result
        assert len(result["should"]) == 2


class TestBuildScrollFilter:
    """Tests for building scroll filter from SQL query."""

    def test_filter_includes_table(self):
        """Test that filter always includes table."""
        query = SQLQuery(table="employees", select=["COUNT(*)"], where=[])
        result = _build_scroll_filter(query)

        assert result["key"] == FIELD_TABLE
        assert result["match"]["value"] == "employees"

    def test_filter_with_where_conditions(self):
        """Test filter with WHERE conditions."""
        query = SQLQuery(
            table="employees",
            select=["COUNT(*)"],
            where=[
                {"column": "department", "op": "=", "value": "eng"},
                {"column": "salary", "op": ">", "value": 50000},
            ],
        )
        result = _build_scroll_filter(query)

        assert "must" in result
        assert len(result["must"]) == 3  # table + 2 conditions


class TestAggregations:
    """Tests for aggregation functions."""

    @pytest.fixture
    def sample_rows(self) -> list[dict[str, Any]]:
        """Create sample rows for testing."""
        return [
            {FIELD_ROW_DATA: {"name": "Alice", "salary": 100000, "department": "eng"}},
            {FIELD_ROW_DATA: {"name": "Bob", "salary": 80000, "department": "eng"}},
            {FIELD_ROW_DATA: {"name": "Charlie", "salary": 120000, "department": "sales"}},
            {FIELD_ROW_DATA: {"name": "Diana", "salary": 90000, "department": "eng"}},
        ]

    def test_count_star(self, sample_rows):
        """Test COUNT(*)."""
        result = _aggregate_count(sample_rows, "*")
        assert result == 4

    def test_count_column(self, sample_rows):
        """Test COUNT(column)."""
        result = _aggregate_count(sample_rows, "salary")
        assert result == 4

    def test_sum(self, sample_rows):
        """Test SUM."""
        result = _aggregate_sum(sample_rows, "salary")
        assert result == 390000

    def test_avg(self, sample_rows):
        """Test AVG."""
        result = _aggregate_avg(sample_rows, "salary")
        assert result == 97500

    def test_min(self, sample_rows):
        """Test MIN."""
        result = _aggregate_min(sample_rows, "salary")
        assert result == 80000

    def test_max(self, sample_rows):
        """Test MAX."""
        result = _aggregate_max(sample_rows, "salary")
        assert result == 120000

    def test_group_concat(self, sample_rows):
        """Test GROUP_CONCAT."""
        result = _aggregate_group_concat(sample_rows, "name")
        assert "Alice" in result
        assert "Bob" in result
        assert "Charlie" in result
        assert "Diana" in result

    def test_group_concat_limit(self, sample_rows):
        """Test GROUP_CONCAT with limit."""
        result = _aggregate_group_concat(sample_rows, "name", limit=2)
        assert "Alice" in result
        assert "Bob" in result
        assert "2 more" in result


class TestStructuredExecutor:
    """Tests for StructuredExecutor."""

    @pytest.fixture
    def sample_data(self) -> list[dict[str, Any]]:
        """Create sample employee data."""
        return [
            {
                FIELD_TABLE: "employees",
                "department": "eng",
                "salary": 100000,
                FIELD_ROW_DATA: {"name": "Alice", "department": "eng", "salary": 100000},
            },
            {
                FIELD_TABLE: "employees",
                "department": "eng",
                "salary": 80000,
                FIELD_ROW_DATA: {"name": "Bob", "department": "eng", "salary": 80000},
            },
            {
                FIELD_TABLE: "employees",
                "department": "sales",
                "salary": 120000,
                FIELD_ROW_DATA: {"name": "Charlie", "department": "sales", "salary": 120000},
            },
        ]

    def test_execute_count(self, sample_data):
        """Test executing COUNT query."""
        db = MockVectorDBClient(sample_data)
        executor = StructuredExecutor(db, "test")

        query = SQLQuery(
            table="employees",
            select=["COUNT(*)"],
            where=[],
        )

        result = executor.execute(query)

        assert result.is_success
        assert result.data["COUNT(*)"] == 3
        assert result.row_count == 3

    def test_execute_sum(self, sample_data):
        """Test executing SUM query."""
        db = MockVectorDBClient(sample_data)
        executor = StructuredExecutor(db, "test")

        query = SQLQuery(
            table="employees",
            select=["SUM(salary)"],
            where=[],
        )

        result = executor.execute(query)

        assert result.is_success
        assert result.data["SUM(salary)"] == 300000

    def test_execute_with_filter(self, sample_data):
        """Test executing query with WHERE filter."""
        db = MockVectorDBClient(sample_data)
        executor = StructuredExecutor(db, "test")

        query = SQLQuery(
            table="employees",
            select=["COUNT(*)"],
            where=[{"column": "department", "op": "=", "value": "eng"}],
        )

        result = executor.execute(query)

        # Note: Our mock doesn't actually filter, but we verify the filter is passed
        assert result.is_success
        assert len(db.scroll_calls) >= 1
        assert db.scroll_calls[0]["filter"] is not None

    def test_execute_avg(self, sample_data):
        """Test executing AVG query."""
        db = MockVectorDBClient(sample_data)
        executor = StructuredExecutor(db, "test")

        query = SQLQuery(
            table="employees",
            select=["AVG(salary)"],
            where=[],
        )

        result = executor.execute(query)

        assert result.is_success
        assert result.data["AVG(salary)"] == 100000  # (100k + 80k + 120k) / 3

    def test_execute_multiple_aggregations(self, sample_data):
        """Test executing multiple aggregations."""
        db = MockVectorDBClient(sample_data)
        executor = StructuredExecutor(db, "test")

        query = SQLQuery(
            table="employees",
            select=["COUNT(*)", "SUM(salary)", "AVG(salary)"],
            where=[],
        )

        result = executor.execute(query)

        assert result.is_success
        assert result.data["COUNT(*)"] == 3
        assert result.data["SUM(salary)"] == 300000
        assert result.data["AVG(salary)"] == 100000

    def test_execute_group_by(self, sample_data):
        """Test executing GROUP BY query."""
        db = MockVectorDBClient(sample_data)
        executor = StructuredExecutor(db, "test")

        query = SQLQuery(
            table="employees",
            select=["COUNT(*)"],
            where=[],
            group_by=["department"],
        )

        result = executor.execute(query)

        assert result.is_success
        assert "groups" in result.data
        assert "group_by" in result.data

    def test_execute_empty_result(self):
        """Test executing query with no matching rows."""
        db = MockVectorDBClient([])  # No data
        executor = StructuredExecutor(db, "test")

        query = SQLQuery(
            table="employees",
            select=["COUNT(*)"],
            where=[],
        )

        result = executor.execute(query)

        assert result.is_success
        assert result.data["COUNT(*)"] == 0
        assert result.row_count == 0


class TestQueryLimitExceededError:
    """Tests for QueryLimitExceededError."""

    def test_error_message(self):
        """Test error message format."""
        error = QueryLimitExceededError(15000, 10000)

        assert "15000" in str(error)
        assert "10000" in str(error)
        assert error.scanned == 15000
        assert error.limit == 10000
