# tests/unit/structured/test_sql_generator.py
"""Tests for SQL generation."""

from __future__ import annotations

import json
from typing import Any

import pytest

from fitz_ai.structured.schema import ColumnSchema, TableSchema
from fitz_ai.structured.sql_generator import (
    SQLGenerator,
    SQLQuery,
    _parse_group_by,
    _parse_limit,
    _parse_order_by,
    _parse_select_clause,
    _parse_where_clause,
)


class MockChatClient:
    """Mock chat client for SQL generation."""

    def __init__(self, response: dict[str, Any] | None = None):
        self.response = response
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]]) -> str:
        self.calls.append(messages)
        if self.response:
            return json.dumps(self.response)
        # Default response
        return json.dumps({
            "queries": [
                {
                    "sql": "SELECT COUNT(*) FROM employees WHERE department = 'engineering'",
                    "table": "employees",
                    "description": "Count engineering employees"
                }
            ]
        })


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
        ],
        primary_key="id",
        row_count=100,
    )


class TestParseSelectClause:
    """Tests for SELECT clause parsing."""

    def test_parse_simple_columns(self):
        """Test parsing simple column names."""
        sql = "SELECT name, salary FROM employees"
        result = _parse_select_clause(sql)
        assert result == ["name", "salary"]

    def test_parse_count_star(self):
        """Test parsing COUNT(*)."""
        sql = "SELECT COUNT(*) FROM employees"
        result = _parse_select_clause(sql)
        assert result == ["COUNT(*)"]

    def test_parse_aggregation_with_column(self):
        """Test parsing aggregation with column name."""
        sql = "SELECT SUM(salary) FROM employees"
        result = _parse_select_clause(sql)
        assert result == ["SUM(salary)"]

    def test_parse_multiple_aggregations(self):
        """Test parsing multiple aggregations."""
        sql = "SELECT COUNT(*), AVG(salary), MAX(salary) FROM employees"
        result = _parse_select_clause(sql)
        assert len(result) == 3
        assert "COUNT(*)" in result
        assert "AVG(salary)" in result
        assert "MAX(salary)" in result

    def test_parse_group_concat(self):
        """Test parsing GROUP_CONCAT."""
        sql = "SELECT GROUP_CONCAT(name, ', ') FROM employees"
        result = _parse_select_clause(sql)
        assert len(result) == 1
        assert "GROUP_CONCAT" in result[0]


class TestParseWhereClause:
    """Tests for WHERE clause parsing."""

    def test_parse_equality(self):
        """Test parsing equality condition."""
        sql = "SELECT * FROM employees WHERE department = 'engineering'"
        result = _parse_where_clause(sql)

        assert len(result) == 1
        assert result[0]["column"] == "department"
        assert result[0]["op"] == "="
        assert result[0]["value"] == "engineering"

    def test_parse_comparison(self):
        """Test parsing comparison operators."""
        sql = "SELECT * FROM employees WHERE salary > 100000"
        result = _parse_where_clause(sql)

        assert len(result) == 1
        assert result[0]["column"] == "salary"
        assert result[0]["op"] == ">"
        assert result[0]["value"] == 100000

    def test_parse_multiple_conditions(self):
        """Test parsing multiple AND conditions."""
        sql = "SELECT * FROM employees WHERE department = 'eng' AND salary > 50000"
        result = _parse_where_clause(sql)

        assert len(result) == 2

    def test_parse_between(self):
        """Test parsing BETWEEN condition."""
        sql = "SELECT * FROM employees WHERE salary BETWEEN 50000 AND 100000"
        result = _parse_where_clause(sql)

        assert len(result) == 1
        assert result[0]["op"] == "BETWEEN"
        assert result[0]["value"] == [50000, 100000]

    def test_parse_in(self):
        """Test parsing IN condition."""
        sql = "SELECT * FROM employees WHERE department IN ('eng', 'sales', 'hr')"
        result = _parse_where_clause(sql)

        assert len(result) == 1
        assert result[0]["op"] == "IN"
        assert "eng" in result[0]["value"]

    def test_parse_no_where(self):
        """Test parsing SQL without WHERE clause."""
        sql = "SELECT COUNT(*) FROM employees"
        result = _parse_where_clause(sql)
        assert result == []


class TestParseGroupBy:
    """Tests for GROUP BY parsing."""

    def test_parse_single_column(self):
        """Test parsing single GROUP BY column."""
        sql = "SELECT department, COUNT(*) FROM employees GROUP BY department"
        result = _parse_group_by(sql)

        assert result == ["department"]

    def test_parse_multiple_columns(self):
        """Test parsing multiple GROUP BY columns."""
        sql = "SELECT dept, level, COUNT(*) FROM emp GROUP BY dept, level"
        result = _parse_group_by(sql)

        assert result == ["dept", "level"]

    def test_parse_no_group_by(self):
        """Test parsing SQL without GROUP BY."""
        sql = "SELECT COUNT(*) FROM employees"
        result = _parse_group_by(sql)
        assert result is None


class TestParseOrderBy:
    """Tests for ORDER BY parsing."""

    def test_parse_desc(self):
        """Test parsing ORDER BY DESC."""
        sql = "SELECT * FROM employees ORDER BY salary DESC"
        column, is_desc = _parse_order_by(sql)

        assert column == "salary"
        assert is_desc is True

    def test_parse_asc(self):
        """Test parsing ORDER BY ASC."""
        sql = "SELECT * FROM employees ORDER BY name ASC"
        column, is_desc = _parse_order_by(sql)

        assert column == "name"
        assert is_desc is False

    def test_parse_default_desc(self):
        """Test that default is DESC."""
        sql = "SELECT * FROM employees ORDER BY salary"
        column, is_desc = _parse_order_by(sql)

        assert column == "salary"
        assert is_desc is True

    def test_parse_no_order_by(self):
        """Test parsing SQL without ORDER BY."""
        sql = "SELECT * FROM employees"
        column, is_desc = _parse_order_by(sql)
        assert column is None


class TestParseLimit:
    """Tests for LIMIT parsing."""

    def test_parse_limit(self):
        """Test parsing LIMIT."""
        sql = "SELECT * FROM employees LIMIT 10"
        result = _parse_limit(sql)
        assert result == 10

    def test_parse_no_limit(self):
        """Test parsing SQL without LIMIT."""
        sql = "SELECT * FROM employees"
        result = _parse_limit(sql)
        assert result is None


class TestSQLQuery:
    """Tests for SQLQuery dataclass."""

    def test_is_aggregation_count(self):
        """Test is_aggregation for COUNT."""
        query = SQLQuery(
            table="employees",
            select=["COUNT(*)"],
            where=[],
        )
        assert query.is_aggregation is True
        assert query.aggregation_type == "COUNT"

    def test_is_aggregation_sum(self):
        """Test is_aggregation for SUM."""
        query = SQLQuery(
            table="employees",
            select=["SUM(salary)"],
            where=[],
        )
        assert query.is_aggregation is True
        assert query.aggregation_type == "SUM"

    def test_is_aggregation_false(self):
        """Test is_aggregation for non-aggregation."""
        query = SQLQuery(
            table="employees",
            select=["name", "salary"],
            where=[],
        )
        assert query.is_aggregation is False
        assert query.aggregation_type is None


class TestSQLGenerator:
    """Tests for SQLGenerator class."""

    def test_generate_basic_query(self, employees_schema: TableSchema):
        """Test basic query generation."""
        client = MockChatClient()
        generator = SQLGenerator(client)

        result = generator.generate(
            "How many employees are in engineering?",
            [employees_schema],
        )

        assert result.error is None
        assert len(result.queries) == 1
        assert result.queries[0].table == "employees"

    def test_generate_calls_chat(self, employees_schema: TableSchema):
        """Test that generator calls chat client."""
        client = MockChatClient()
        generator = SQLGenerator(client)

        generator.generate("Count employees", [employees_schema])

        assert len(client.calls) == 1
        prompt = client.calls[0][0]["content"]
        assert "employees" in prompt
        assert "salary" in prompt

    def test_generate_no_schemas(self):
        """Test generation with no schemas."""
        client = MockChatClient()
        generator = SQLGenerator(client)

        result = generator.generate("Count stuff", [])

        assert result.error is not None
        assert "No schemas" in result.error

    def test_generate_multiple_queries(self, employees_schema: TableSchema):
        """Test generating multiple queries."""
        client = MockChatClient(response={
            "queries": [
                {"sql": "SELECT COUNT(*) FROM employees", "table": "employees"},
                {"sql": "SELECT AVG(salary) FROM employees", "table": "employees"},
            ]
        })
        generator = SQLGenerator(client)

        result = generator.generate("Stats about employees", [employees_schema])

        assert len(result.queries) == 2

    def test_generate_handles_invalid_response(self, employees_schema: TableSchema):
        """Test handling of invalid LLM response."""
        client = MockChatClient(response={"invalid": "response"})
        generator = SQLGenerator(client)

        result = generator.generate("Count employees", [employees_schema])

        assert result.error is not None
