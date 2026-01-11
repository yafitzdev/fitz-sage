# tests/tabular/test_query.py
"""Tests for TableQueryStep - table queries at retrieval time."""

import json
from unittest.mock import MagicMock

import pytest

from fitz_ai.core.chunk import Chunk
from fitz_ai.tabular.models import ParsedTable, create_schema_chunk
from fitz_ai.tabular.query import TableQueryStep


class TestTableQueryStep:
    """Tests for TableQueryStep."""

    @pytest.fixture
    def mock_chat(self):
        """Create a mock chat client."""
        return MagicMock()

    @pytest.fixture
    def step(self, mock_chat):
        """Create step with mock chat."""
        return TableQueryStep(chat=mock_chat)

    @pytest.fixture
    def sample_table_chunk(self):
        """Create a sample table schema chunk."""
        table = ParsedTable(
            table_id="test123",
            source_doc="data.csv",
            headers=["Country", "Population", "GDP", "Continent"],
            rows=[
                ["USA", "330000000", "21000", "North America"],
                ["China", "1400000000", "14000", "Asia"],
                ["Germany", "83000000", "3800", "Europe"],
                ["Brazil", "210000000", "1800", "South America"],
            ],
        )
        return create_schema_chunk(table)

    def test_passes_through_non_table_chunks(self, step):
        """Test that non-table chunks pass through unchanged."""
        regular_chunk = Chunk(
            id="regular1",
            doc_id="doc1",
            content="This is regular content",
            chunk_index=0,
            metadata={},
        )

        result = step.execute("any query", [regular_chunk])

        assert len(result) == 1
        assert result[0].id == "regular1"
        assert result[0].content == "This is regular content"

    def test_processes_table_chunk(self, step, mock_chat, sample_table_chunk):
        """Test processing a table schema chunk."""
        # Mock LLM responses
        mock_chat.chat.side_effect = [
            '["Country", "Population"]',  # Column selection
            'SELECT Country, Population FROM data WHERE Population > 100000000',  # SQL
        ]

        result = step.execute("Countries with population over 100 million", [sample_table_chunk])

        assert len(result) == 1
        # Should have augmented content
        assert "Query Results" in result[0].content
        assert "sql_executed" in result[0].metadata

    def test_handles_mixed_chunks(self, step, mock_chat, sample_table_chunk):
        """Test handling mix of table and regular chunks."""
        regular_chunk = Chunk(
            id="reg1",
            doc_id="doc1",
            content="Regular content",
            chunk_index=0,
            metadata={},
        )

        # Mock for table chunk processing
        mock_chat.chat.side_effect = [
            '["Country"]',
            'SELECT Country FROM data',
        ]

        result = step.execute("query", [regular_chunk, sample_table_chunk])

        assert len(result) == 2
        # Table chunks processed first, then regular chunks
        assert "Query Results" in result[0].content  # Table result
        assert result[1].content == "Regular content"  # Regular chunk unchanged

    def test_column_selection_fallback(self, step, mock_chat, sample_table_chunk):
        """Test fallback when column selection returns invalid JSON."""
        mock_chat.chat.side_effect = [
            "invalid json response",  # Bad column selection
            'SELECT * FROM data',  # SQL with all columns
        ]

        result = step.execute("query", [sample_table_chunk])

        # Should still work (fallback to all columns)
        assert len(result) == 1
        # Chat was called twice
        assert mock_chat.chat.call_count == 2

    def test_sql_error_handling(self, step, mock_chat, sample_table_chunk):
        """Test handling of SQL execution errors."""
        mock_chat.chat.side_effect = [
            '["Country"]',
            'SELECT NonExistentColumn FROM data',  # Invalid SQL
        ]

        result = step.execute("query", [sample_table_chunk])

        # Should return chunk with error info
        assert len(result) == 1
        assert "Error:" in result[0].content or "sql_error" in result[0].metadata

    def test_missing_table_data_in_metadata(self, step):
        """Test handling of chunk without table_data."""
        broken_chunk = Chunk(
            id="broken1",
            doc_id="doc1",
            content="Table schema",
            chunk_index=0,
            metadata={"is_table_schema": True},  # Missing table_data
        )

        result = step.execute("query", [broken_chunk])

        # Should return original chunk on error
        assert len(result) == 1
        assert result[0].id == "broken1"


class TestColumnSelection:
    """Tests for column selection logic."""

    @pytest.fixture
    def step(self):
        mock_chat = MagicMock()
        return TableQueryStep(chat=mock_chat), mock_chat

    def test_parses_json_array(self, step):
        """Test parsing valid JSON array response."""
        step_instance, mock_chat = step
        mock_chat.chat.return_value = '["col1", "col2", "col3"]'

        columns = step_instance._select_columns("query", ["col1", "col2", "col3", "col4"])

        assert columns == ["col1", "col2", "col3"]

    def test_handles_markdown_code_block(self, step):
        """Test parsing JSON in markdown code block."""
        step_instance, mock_chat = step
        mock_chat.chat.return_value = '''```json
["Country", "Population"]
```'''

        columns = step_instance._select_columns("query", ["Country", "Population", "GDP"])

        assert columns == ["Country", "Population"]

    def test_fallback_on_invalid_json(self, step):
        """Test fallback when JSON is invalid."""
        step_instance, mock_chat = step
        mock_chat.chat.return_value = "not valid json at all"

        all_columns = ["A", "B", "C"]
        columns = step_instance._select_columns("query", all_columns)

        # Should fallback to all columns
        assert columns == all_columns


class TestSQLGeneration:
    """Tests for SQL generation logic."""

    @pytest.fixture
    def step(self):
        mock_chat = MagicMock()
        return TableQueryStep(chat=mock_chat), mock_chat

    def test_extracts_sql_from_response(self, step):
        """Test extracting SQL from plain response."""
        step_instance, mock_chat = step
        mock_chat.chat.return_value = "SELECT Country FROM data WHERE Population > 1000000"

        sql = step_instance._generate_sql(
            "large countries",
            ["Country", "Population"],
            [["USA", "330000000"]],
        )

        assert sql.upper().startswith("SELECT")

    def test_extracts_sql_from_code_block(self, step):
        """Test extracting SQL from markdown code block."""
        step_instance, mock_chat = step
        mock_chat.chat.return_value = '''Here's the query:
```sql
SELECT Country, Population FROM data WHERE Population > 1000000
```
This will return countries with large populations.'''

        sql = step_instance._generate_sql(
            "large countries",
            ["Country", "Population"],
            [["USA", "330000000"]],
        )

        assert "SELECT" in sql.upper()
        assert "Population" in sql


class TestResultFormatting:
    """Tests for result formatting."""

    @pytest.fixture
    def step(self):
        mock_chat = MagicMock()
        return TableQueryStep(chat=mock_chat)

    def test_formats_results_as_markdown_table(self, step):
        """Test markdown table formatting."""
        result = step._format_as_markdown(
            ["Name", "Value"],
            [("Alice", 100), ("Bob", 200)],
        )

        assert "| Name | Value |" in result
        assert "| --- | --- |" in result
        assert "| Alice | 100 |" in result
        assert "| Bob | 200 |" in result

    def test_handles_empty_results(self, step):
        """Test formatting empty results."""
        result = step._format_as_markdown(["Col"], [])

        assert "(no results)" in result

    def test_truncates_long_values(self, step):
        """Test truncation of long cell values."""
        long_value = "A" * 100
        result = step._format_as_markdown(["Data"], [(long_value,)])

        # Should be truncated
        assert "..." in result
        assert len(result) < len(long_value)

    def test_limits_result_rows(self, step):
        """Test limiting number of displayed rows."""
        step.max_results = 5
        many_rows = [(f"row{i}",) for i in range(100)]

        result = step._format_as_markdown(["ID"], many_rows)

        # Should mention more rows
        assert "more rows" in result


class TestEndToEnd:
    """End-to-end tests for table query flow."""

    def test_full_query_flow(self):
        """Test complete flow from chunk to augmented result."""
        # Create table
        table = ParsedTable(
            table_id="e2e_test",
            source_doc="countries.csv",
            headers=["Country", "Population", "Capital"],
            rows=[
                ["USA", "330000000", "Washington"],
                ["France", "67000000", "Paris"],
                ["Japan", "125000000", "Tokyo"],
            ],
        )
        chunk = create_schema_chunk(table)

        # Setup mock chat
        mock_chat = MagicMock()
        mock_chat.chat.side_effect = [
            '["Country", "Capital"]',  # Column selection
            'SELECT Country, Capital FROM data',  # SQL
        ]

        step = TableQueryStep(chat=mock_chat)
        result = step.execute("What are the capitals?", [chunk])

        assert len(result) == 1
        # Check augmented content
        assert "Query Results" in result[0].content
        assert "SELECT Country, Capital FROM data" in result[0].content
        # Results should include the capitals
        assert "Washington" in result[0].content
        assert "Paris" in result[0].content
        assert "Tokyo" in result[0].content


class TestMultiTableJoins:
    """Tests for multi-table join functionality."""

    @pytest.fixture
    def step(self):
        mock_chat = MagicMock()
        return TableQueryStep(chat=mock_chat), mock_chat

    @pytest.fixture
    def employees_chunk(self):
        """Create employees table chunk."""
        table = ParsedTable(
            table_id="emp123",
            source_doc="employees.csv",
            headers=["id", "name", "dept_id"],
            rows=[
                ["1", "Alice", "10"],
                ["2", "Bob", "20"],
                ["3", "Carol", "10"],
            ],
        )
        return create_schema_chunk(table)

    @pytest.fixture
    def departments_chunk(self):
        """Create departments table chunk."""
        table = ParsedTable(
            table_id="dept456",
            source_doc="departments.csv",
            headers=["id", "dept_name", "location"],
            rows=[
                ["10", "Engineering", "Building A"],
                ["20", "Sales", "Building B"],
            ],
        )
        return create_schema_chunk(table)

    def test_needs_multi_table_returns_false_for_single_chunk(self, step):
        """Test detection returns false for single table."""
        step_instance, mock_chat = step
        table = ParsedTable(
            table_id="t1",
            source_doc="data.csv",
            headers=["col1"],
            rows=[["val"]],
        )
        chunk = create_schema_chunk(table)

        result = step_instance._needs_multi_table("query", [chunk])

        assert result is False
        # LLM should not be called for single table
        mock_chat.chat.assert_not_called()

    def test_needs_multi_table_calls_llm_for_multiple_chunks(
        self, step, employees_chunk, departments_chunk
    ):
        """Test detection calls LLM for multiple tables."""
        step_instance, mock_chat = step
        mock_chat.chat.return_value = "no"

        result = step_instance._needs_multi_table(
            "What is Alice's name?",
            [employees_chunk, departments_chunk],
        )

        assert result is False
        mock_chat.chat.assert_called_once()

    def test_needs_multi_table_returns_true_for_join_query(
        self, step, employees_chunk, departments_chunk
    ):
        """Test detection returns true for join query."""
        step_instance, mock_chat = step
        mock_chat.chat.return_value = "yes"

        result = step_instance._needs_multi_table(
            "Show employee names with their department names",
            [employees_chunk, departments_chunk],
        )

        assert result is True

    def test_derive_table_name_from_doc_id(self, step):
        """Test table name derivation from doc_id."""
        step_instance, _ = step

        chunk = Chunk(
            id="test",
            doc_id="employees.csv",
            content="",
            chunk_index=0,
            metadata={"table_id": "abc123"},
        )

        name = step_instance._derive_table_name(chunk)

        assert name == "employees"

    def test_derive_table_name_sanitizes_special_chars(self, step):
        """Test table name sanitization."""
        step_instance, _ = step

        chunk = Chunk(
            id="test",
            doc_id="my-data (2024).csv",
            content="",
            chunk_index=0,
            metadata={"table_id": "abc123"},
        )

        name = step_instance._derive_table_name(chunk)

        # Special chars replaced with underscores
        assert "-" not in name
        assert "(" not in name
        assert ")" not in name
        assert " " not in name

    def test_derive_table_name_handles_leading_digit(self, step):
        """Test table name with leading digit gets prefix."""
        step_instance, _ = step

        chunk = Chunk(
            id="test",
            doc_id="2024_data.csv",
            content="",
            chunk_index=0,
            metadata={"table_id": "abc123"},
        )

        name = step_instance._derive_table_name(chunk)

        # Should prefix with t_
        assert name.startswith("t_")

    def test_multi_table_execution(self, step, employees_chunk, departments_chunk):
        """Test multi-table query execution."""
        step_instance, mock_chat = step

        # Mock responses: detection=yes, SQL generation
        mock_chat.chat.side_effect = [
            "yes",  # Multi-table detection
            """SELECT e.name, d.dept_name
               FROM employees e
               JOIN departments d ON e.dept_id = d.id""",  # SQL
        ]

        result = step_instance.execute(
            "Show employee names with department names",
            [employees_chunk, departments_chunk],
        )

        assert len(result) == 1
        assert result[0].metadata.get("is_multi_table_result") is True
        assert "employees" in result[0].metadata.get("source_tables", [])
        assert "departments" in result[0].metadata.get("source_tables", [])

    def test_multi_table_fallback_on_single_table_query(
        self, step, employees_chunk, departments_chunk
    ):
        """Test fallback to single-table processing when not a join query."""
        step_instance, mock_chat = step

        # Mock: detection=no, then single table processing
        mock_chat.chat.side_effect = [
            "no",  # Multi-table detection - not needed
            '["name"]',  # Column selection
            "SELECT name FROM data",  # SQL
            '["dept_name"]',  # Column selection for second table
            "SELECT dept_name FROM data",  # SQL for second table
        ]

        result = step_instance.execute(
            "What are the employee names?",
            [employees_chunk, departments_chunk],
        )

        # Should process each table independently
        assert len(result) == 2

    def test_multi_table_sql_generation(self, step):
        """Test multi-table SQL generation prompt."""
        step_instance, mock_chat = step
        mock_chat.chat.return_value = """SELECT p.name, s.company
FROM products p JOIN suppliers s ON p.supplier_id = s.id"""

        tables = [
            ("products", ["id", "name", "supplier_id"], [["1", "Widget", "100"]]),
            ("suppliers", ["id", "company"], [["100", "Acme Corp"]]),
        ]

        sql = step_instance._generate_multi_table_sql("products with suppliers", tables)

        assert "SELECT" in sql.upper()
        assert "JOIN" in sql.upper()

    def test_create_and_populate_named(self, step):
        """Test creating named table in SQLite."""
        step_instance, _ = step
        import sqlite3

        conn = sqlite3.connect(":memory:")
        step_instance._create_and_populate_named(
            conn,
            "test_table",
            ["col1", "col2"],
            [["a", "1"], ["b", "2"]],
        )

        # Verify table exists with correct data
        cursor = conn.execute("SELECT * FROM test_table")
        rows = cursor.fetchall()

        assert len(rows) == 2
        assert rows[0] == ("a", "1")
        assert rows[1] == ("b", "2")

        conn.close()
