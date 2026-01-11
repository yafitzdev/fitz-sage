# tests/tabular/test_models.py
"""Tests for tabular data models."""

import json


from fitz_ai.tabular.models import ParsedTable, create_schema_chunk


class TestParsedTable:
    """Tests for ParsedTable dataclass."""

    def test_create_simple_table(self):
        """Test creating a simple table."""
        table = ParsedTable(
            table_id="test123",
            source_doc="test.pdf",
            headers=["Name", "Age", "City"],
            rows=[
                ["Alice", "30", "NYC"],
                ["Bob", "25", "LA"],
            ],
        )

        assert table.table_id == "test123"
        assert table.source_doc == "test.pdf"
        assert table.headers == ["Name", "Age", "City"]
        assert table.row_count == 2
        assert table.column_count == 3

    def test_to_json_serialization(self):
        """Test JSON serialization."""
        table = ParsedTable(
            table_id="test123",
            source_doc="test.pdf",
            headers=["A", "B"],
            rows=[["1", "2"], ["3", "4"]],
        )

        json_str = table.to_json()
        parsed = json.loads(json_str)

        assert parsed["headers"] == ["A", "B"]
        assert parsed["rows"] == [["1", "2"], ["3", "4"]]

    def test_from_json_deserialization(self):
        """Test JSON deserialization."""
        json_str = json.dumps(
            {
                "headers": ["X", "Y", "Z"],
                "rows": [["a", "b", "c"], ["d", "e", "f"]],
            }
        )

        table = ParsedTable.from_json(json_str, "restored123", "restored.pdf")

        assert table.table_id == "restored123"
        assert table.source_doc == "restored.pdf"
        assert table.headers == ["X", "Y", "Z"]
        assert table.rows == [["a", "b", "c"], ["d", "e", "f"]]

    def test_roundtrip_serialization(self):
        """Test JSON round-trip."""
        original = ParsedTable(
            table_id="round123",
            source_doc="round.pdf",
            headers=["Col1", "Col2"],
            rows=[["val1", "val2"]],
        )

        json_str = original.to_json()
        restored = ParsedTable.from_json(json_str, original.table_id, original.source_doc)

        assert restored.headers == original.headers
        assert restored.rows == original.rows


class TestCreateSchemaChunk:
    """Tests for schema chunk creation."""

    def test_creates_chunk_with_correct_id(self):
        """Test chunk ID format."""
        table = ParsedTable(
            table_id="abc123",
            source_doc="doc.pdf",
            headers=["A"],
            rows=[["1"]],
        )

        chunk = create_schema_chunk(table)

        assert chunk.id == "table_abc123"

    def test_includes_table_data_in_metadata(self):
        """Test that full table data is in metadata."""
        table = ParsedTable(
            table_id="data123",
            source_doc="data.pdf",
            headers=["X", "Y"],
            rows=[["1", "2"], ["3", "4"]],
        )

        chunk = create_schema_chunk(table)

        assert chunk.metadata["is_table_schema"] is True
        assert chunk.metadata["table_id"] == "data123"
        assert chunk.metadata["columns"] == ["X", "Y"]
        assert chunk.metadata["row_count"] == 2
        assert "table_data" in chunk.metadata

        # Verify table_data can be deserialized
        restored_data = json.loads(chunk.metadata["table_data"])
        assert restored_data["headers"] == ["X", "Y"]
        assert restored_data["rows"] == [["1", "2"], ["3", "4"]]

    def test_content_is_human_readable(self):
        """Test that content is human-readable for embedding."""
        table = ParsedTable(
            table_id="readable123",
            source_doc="report.pdf",
            headers=["Product", "Price", "Quantity"],
            rows=[
                ["Widget", "9.99", "100"],
                ["Gadget", "19.99", "50"],
            ],
        )

        chunk = create_schema_chunk(table)

        assert "Table from report.pdf" in chunk.content
        assert "Columns: Product, Price, Quantity" in chunk.content
        assert "Row count: 2 rows" in chunk.content
        assert "Widget" in chunk.content

    def test_sample_data_truncation(self):
        """Test that sample data is truncated for large tables."""
        # Create table with many rows
        table = ParsedTable(
            table_id="big123",
            source_doc="big.pdf",
            headers=["ID"],
            rows=[[str(i)] for i in range(100)],
        )

        chunk = create_schema_chunk(table)

        # Should show only first 3 rows + "more rows" message
        assert "... and 97 more rows" in chunk.content

    def test_long_cell_truncation(self):
        """Test that long cell values are truncated in sample."""
        table = ParsedTable(
            table_id="long123",
            source_doc="long.pdf",
            headers=["Content"],
            rows=[
                [
                    "This is a very long cell value that should be truncated to prevent overly long content"
                ]
            ],
        )

        chunk = create_schema_chunk(table)

        # Content should be truncated
        assert "..." in chunk.content
