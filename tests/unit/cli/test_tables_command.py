# tests/unit/cli/test_tables_command.py
"""
Tests for the tables CLI command module.
"""

import csv
from unittest.mock import MagicMock, patch

from fitz_ai.structured.schema import ColumnSchema, TableSchema


class TestReadCSV:
    """Tests for CSV reading functionality."""

    def test_read_csv_simple(self, tmp_path):
        """Test reading a simple CSV file."""
        from fitz_ai.cli.commands.tables import _read_csv

        # Create test CSV
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "value"])
            writer.writerow(["1", "Alice", "100"])
            writer.writerow(["2", "Bob", "200"])

        headers, rows = _read_csv(csv_path)

        assert headers == ["id", "name", "value"]
        assert len(rows) == 2
        assert rows[0]["id"] == "1"
        assert rows[0]["name"] == "Alice"
        assert rows[1]["value"] == "200"

    def test_read_csv_with_bom(self, tmp_path):
        """Test reading a CSV with BOM marker."""
        from fitz_ai.cli.commands.tables import _read_csv

        csv_path = tmp_path / "test_bom.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name"])
            writer.writerow(["1", "Test"])

        headers, rows = _read_csv(csv_path)

        # Headers should not include BOM
        assert headers == ["id", "name"]
        assert len(rows) == 1


class TestDisplayFunctions:
    """Tests for display helper functions."""

    def test_display_tables_list_empty(self, capsys):
        """Test displaying empty table list."""
        from fitz_ai.cli.commands.tables import _display_tables_list

        _display_tables_list([])

        # Should not raise, just print nothing meaningful


class TestTableSchemaDisplay:
    """Tests for table schema display."""

    def test_display_table_schema(self, capsys):
        """Test displaying table schema."""
        from fitz_ai.cli.commands.tables import _display_table_schema

        schema = TableSchema(
            table_name="employees",
            columns=[
                ColumnSchema(name="id", type="number", indexed=True, nullable=False),
                ColumnSchema(name="name", type="string", indexed=False, nullable=False),
                ColumnSchema(name="salary", type="number", indexed=True, nullable=True),
            ],
            primary_key="id",
            row_count=100,
        )

        # Should not raise
        _display_table_schema(schema)

        captured = capsys.readouterr()
        assert "employees" in captured.out
        assert "100" in captured.out or "Rows" in captured.out


class TestGetCollection:
    """Tests for collection name resolution."""

    def test_get_collection_with_value(self):
        """Test getting collection when value is provided."""
        from fitz_ai.cli.commands.tables import _get_collection

        result = _get_collection("my_collection")
        assert result == "my_collection"

    @patch("fitz_ai.cli.context.CLIContext")
    def test_get_collection_from_context(self, mock_context_class):
        """Test getting collection from CLI context."""
        from fitz_ai.cli.commands.tables import _get_collection

        mock_ctx = MagicMock()
        mock_ctx.retrieval_collection = "default_collection"
        mock_context_class.load.return_value = mock_ctx

        result = _get_collection(None)
        assert result == "default_collection"


class TestPrimaryKeyDetection:
    """Tests for primary key auto-detection."""

    def test_detect_pk_by_name_pattern(self):
        """Test PK detection based on column name patterns."""
        # This is tested implicitly in ingest_table_command
        # but we can verify the patterns work
        pk_patterns = ["id", "_id", "key", "_key", "pk", "primary"]

        test_columns = [
            ("user_id", True),
            ("employee_id", True),
            ("primary_key", True),
            ("record_pk", True),
            ("name", False),
            ("value", False),
        ]

        for col, should_match in test_columns:
            col_lower = col.lower()
            matched = any(p in col_lower for p in pk_patterns)
            assert matched == should_match, f"Column {col} should match={should_match}"


class TestCLIRegistration:
    """Tests for CLI command registration."""

    def test_tables_app_exists(self):
        """Test that tables app is properly defined."""
        from fitz_ai.cli.commands.tables import app

        assert app is not None
        assert app.info.name == "tables"

    def test_ingest_table_command_exists(self):
        """Test that ingest_table_command function exists."""
        from fitz_ai.cli.commands.tables import ingest_table_command

        assert callable(ingest_table_command)

    def test_list_command_exists(self):
        """Test that list command exists."""
        from fitz_ai.cli.commands.tables import list_tables

        assert callable(list_tables)

    def test_info_command_exists(self):
        """Test that info command exists."""
        from fitz_ai.cli.commands.tables import table_info

        assert callable(table_info)

    def test_delete_command_exists(self):
        """Test that delete command exists."""
        from fitz_ai.cli.commands.tables import delete_table

        assert callable(delete_table)


class TestMainCLIIntegration:
    """Tests for main CLI integration."""

    def test_tables_registered_in_main_cli(self):
        """Test that tables is registered in main CLI."""
        from fitz_ai.cli.cli import app

        # Check that tables subcommand is registered
        command_names = [cmd.name for cmd in app.registered_groups]
        assert "tables" in command_names

    def test_ingest_table_registered_in_main_cli(self):
        """Test that ingest-table is registered in main CLI."""
        from fitz_ai.cli.cli import app

        # Check that ingest-table command is registered
        command_names = [cmd.name or cmd.callback.__name__ for cmd in app.registered_commands]
        assert "ingest-table" in command_names


class TestSchemaStoreIntegration:
    """Tests for SchemaStore method additions."""

    def test_get_schema_method_exists(self):
        """Test that get_schema method exists on SchemaStore."""
        from fitz_ai.structured.schema import SchemaStore

        assert hasattr(SchemaStore, "get_schema")

    def test_get_all_schemas_method_exists(self):
        """Test that get_all_schemas method exists on SchemaStore."""
        from fitz_ai.structured.schema import SchemaStore

        assert hasattr(SchemaStore, "get_all_schemas")

    def test_delete_schema_method_exists(self):
        """Test that delete_schema method exists on SchemaStore."""
        from fitz_ai.structured.schema import SchemaStore

        assert hasattr(SchemaStore, "delete_schema")
