# tests/unit/structured/test_formatter.py
"""Tests for result formatting."""

from __future__ import annotations

from typing import Any

import pytest

from fitz_ai.structured.executor import ExecutionResult
from fitz_ai.structured.formatter import (
    FormattedResult,
    ResultFormatter,
    format_multiple_results,
)
from fitz_ai.structured.sql_generator import SQLQuery


class MockChatClient:
    """Mock chat client for formatting."""

    def __init__(self, response: str | None = None):
        self.response = response
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]]) -> str:
        self.calls.append(messages)
        if self.response:
            return self.response
        # Default: extract data from prompt and generate basic sentence
        prompt = messages[0]["content"] if messages else ""
        if "COUNT" in prompt:
            return "There are 42 employees."
        if "SUM" in prompt:
            return "The total salary is $5,000,000."
        return "Query executed successfully."


@pytest.fixture
def count_query() -> SQLQuery:
    """Create a COUNT query."""
    return SQLQuery(
        table="employees",
        select=["COUNT(*)"],
        where=[{"column": "department", "op": "=", "value": "engineering"}],
        raw_sql="SELECT COUNT(*) FROM employees WHERE department = 'engineering'",
    )


@pytest.fixture
def sum_query() -> SQLQuery:
    """Create a SUM query."""
    return SQLQuery(
        table="employees",
        select=["SUM(salary)"],
        where=[],
        raw_sql="SELECT SUM(salary) FROM employees",
    )


@pytest.fixture
def multi_agg_query() -> SQLQuery:
    """Create a query with multiple aggregations."""
    return SQLQuery(
        table="employees",
        select=["COUNT(*)", "SUM(salary)", "AVG(salary)"],
        where=[],
        raw_sql="SELECT COUNT(*), SUM(salary), AVG(salary) FROM employees",
    )


class TestResultFormatter:
    """Tests for ResultFormatter class."""

    def test_format_count_result(self, count_query: SQLQuery):
        """Test formatting COUNT result."""
        client = MockChatClient("There are 42 employees in the engineering department.")
        formatter = ResultFormatter(client)

        execution_result = ExecutionResult(
            data={"COUNT(*)": 42},
            row_count=42,
            query=count_query,
        )

        result = formatter.format(execution_result)

        assert isinstance(result, FormattedResult)
        assert result.sentence == "There are 42 employees in the engineering department."
        assert result.table == "employees"
        assert result.data == {"COUNT(*)": 42}

    def test_format_sum_result(self, sum_query: SQLQuery):
        """Test formatting SUM result."""
        client = MockChatClient("The total salary is $5,000,000.")
        formatter = ResultFormatter(client)

        execution_result = ExecutionResult(
            data={"SUM(salary)": 5000000},
            row_count=100,
            query=sum_query,
        )

        result = formatter.format(execution_result)

        assert "5,000,000" in result.sentence or "5000000" in result.sentence

    def test_format_calls_chat_client(self, count_query: SQLQuery):
        """Test that formatter calls chat client."""
        client = MockChatClient("Test response.")
        formatter = ResultFormatter(client)

        execution_result = ExecutionResult(
            data={"COUNT(*)": 10},
            row_count=10,
            query=count_query,
        )

        formatter.format(execution_result)

        assert len(client.calls) == 1
        prompt = client.calls[0][0]["content"]
        assert "SELECT COUNT(*)" in prompt
        assert "employees" in prompt

    def test_format_includes_raw_sql(self, count_query: SQLQuery):
        """Test that result includes original SQL."""
        client = MockChatClient("Test.")
        formatter = ResultFormatter(client)

        execution_result = ExecutionResult(
            data={"COUNT(*)": 5},
            row_count=5,
            query=count_query,
        )

        result = formatter.format(execution_result)

        assert "SELECT COUNT(*)" in result.query

    def test_format_reconstructs_sql_when_no_raw(self):
        """Test SQL reconstruction when raw_sql is empty."""
        client = MockChatClient("There are 5 items.")
        formatter = ResultFormatter(client)

        query = SQLQuery(
            table="items",
            select=["COUNT(*)"],
            where=[{"column": "status", "op": "=", "value": "active"}],
            raw_sql="",  # No raw SQL
        )

        execution_result = ExecutionResult(
            data={"COUNT(*)": 5},
            row_count=5,
            query=query,
        )

        result = formatter.format(execution_result)

        assert "SELECT" in result.query
        assert "COUNT(*)" in result.query
        assert "items" in result.query
        assert "status" in result.query

    def test_format_handles_group_by(self):
        """Test formatting GROUP BY results."""
        client = MockChatClient("Engineering has 20 employees, sales has 15.")
        formatter = ResultFormatter(client)

        query = SQLQuery(
            table="employees",
            select=["department", "COUNT(*)"],
            where=[],
            group_by=["department"],
            raw_sql="SELECT department, COUNT(*) FROM employees GROUP BY department",
        )

        execution_result = ExecutionResult(
            data={
                "groups": {
                    "engineering": {"COUNT(*)": 20},
                    "sales": {"COUNT(*)": 15},
                },
                "group_by": ["department"],
            },
            row_count=35,
            query=query,
        )

        result = formatter.format(execution_result)

        assert "engineering" in result.sentence.lower() or "20" in result.sentence

    def test_format_strips_quotes(self, count_query: SQLQuery):
        """Test that formatter strips quotes from response."""
        client = MockChatClient('"The count is 10."')
        formatter = ResultFormatter(client)

        execution_result = ExecutionResult(
            data={"COUNT(*)": 10},
            row_count=10,
            query=count_query,
        )

        result = formatter.format(execution_result)

        assert not result.sentence.startswith('"')
        assert not result.sentence.endswith('"')


class TestFallbackFormatting:
    """Tests for fallback formatting when LLM fails."""

    def test_fallback_on_exception(self):
        """Test fallback formatting when chat client raises exception."""

        class FailingClient:
            def chat(self, messages):
                raise RuntimeError("LLM unavailable")

        formatter = ResultFormatter(FailingClient())

        query = SQLQuery(
            table="items",
            select=["COUNT(*)"],
            where=[],
            raw_sql="SELECT COUNT(*) FROM items",
        )

        execution_result = ExecutionResult(
            data={"COUNT(*)": 25},
            row_count=25,
            query=query,
        )

        result = formatter.format(execution_result)

        assert "count" in result.sentence.lower()
        assert "25" in result.sentence

    def test_fallback_sum_formatting(self):
        """Test fallback formatting for SUM."""

        class FailingClient:
            def chat(self, messages):
                raise RuntimeError("LLM unavailable")

        formatter = ResultFormatter(FailingClient())

        query = SQLQuery(
            table="sales",
            select=["SUM(revenue)"],
            where=[],
            raw_sql="SELECT SUM(revenue) FROM sales",
        )

        execution_result = ExecutionResult(
            data={"SUM(revenue)": 1000000},
            row_count=100,
            query=query,
        )

        result = formatter.format(execution_result)

        assert "total" in result.sentence.lower()
        assert "1,000,000" in result.sentence or "1000000" in result.sentence

    def test_fallback_avg_formatting(self):
        """Test fallback formatting for AVG."""

        class FailingClient:
            def chat(self, messages):
                raise RuntimeError("LLM unavailable")

        formatter = ResultFormatter(FailingClient())

        query = SQLQuery(
            table="products",
            select=["AVG(price)"],
            where=[],
            raw_sql="SELECT AVG(price) FROM products",
        )

        execution_result = ExecutionResult(
            data={"AVG(price)": 49.99},
            row_count=50,
            query=query,
        )

        result = formatter.format(execution_result)

        assert "average" in result.sentence.lower()

    def test_fallback_empty_result(self):
        """Test fallback formatting for empty result."""

        class FailingClient:
            def chat(self, messages):
                raise RuntimeError("LLM unavailable")

        formatter = ResultFormatter(FailingClient())

        query = SQLQuery(
            table="empty_table",
            select=["COUNT(*)"],
            where=[],
            raw_sql="SELECT COUNT(*) FROM empty_table",
        )

        execution_result = ExecutionResult(
            data={},
            row_count=0,
            query=query,
        )

        result = formatter.format(execution_result)

        assert "no results" in result.sentence.lower()


class TestFormatMultipleResults:
    """Tests for format_multiple_results helper."""

    def test_format_multiple(self):
        """Test formatting multiple results."""
        client = MockChatClient("Result.")
        formatter = ResultFormatter(client)

        results = [
            ExecutionResult(
                data={"COUNT(*)": 10},
                row_count=10,
                query=SQLQuery(table="t1", select=["COUNT(*)"], where=[]),
            ),
            ExecutionResult(
                data={"SUM(x)": 100},
                row_count=5,
                query=SQLQuery(table="t2", select=["SUM(x)"], where=[]),
            ),
        ]

        formatted = format_multiple_results(formatter, results)

        assert len(formatted) == 2
        assert all(isinstance(r, FormattedResult) for r in formatted)

    def test_format_empty_list(self):
        """Test formatting empty list."""
        client = MockChatClient()
        formatter = ResultFormatter(client)

        formatted = format_multiple_results(formatter, [])

        assert formatted == []


class TestFormattedResult:
    """Tests for FormattedResult dataclass."""

    def test_formatted_result_fields(self):
        """Test FormattedResult has all expected fields."""
        result = FormattedResult(
            sentence="There are 42 employees.",
            query="SELECT COUNT(*) FROM employees",
            table="employees",
            data={"COUNT(*)": 42},
        )

        assert result.sentence == "There are 42 employees."
        assert result.query == "SELECT COUNT(*) FROM employees"
        assert result.table == "employees"
        assert result.data == {"COUNT(*)": 42}
