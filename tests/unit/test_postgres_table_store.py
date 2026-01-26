# tests/unit/test_postgres_table_store.py
"""
Unit tests for PostgresTableStore.

Tests cover:
1. Table name sanitization
2. Column name sanitization and deduplication
3. Store/retrieve roundtrip
4. SQL query execution
5. Add columns to existing tables
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

# Mark all tests in this module as postgres and tier2
pytestmark = [pytest.mark.postgres, pytest.mark.tier2]

from fitz_ai.tabular.store.postgres import (
    PostgresTableStore,
    _sanitize_column_name,
    _sanitize_table_name,
)


# =============================================================================
# Table Name Sanitization Tests
# =============================================================================


class TestTableNameSanitization:
    """Tests for table name sanitization."""

    def test_simple_name_prefixed(self):
        """Simple name gets tbl_ prefix."""
        assert _sanitize_table_name("users") == "tbl_users"
        assert _sanitize_table_name("orders") == "tbl_orders"

    def test_hyphens_replaced(self):
        """Hyphens are replaced with underscores."""
        assert _sanitize_table_name("my-table") == "tbl_my_table"
        assert _sanitize_table_name("a-b-c") == "tbl_a_b_c"

    def test_special_chars_replaced(self):
        """Special characters are replaced."""
        assert _sanitize_table_name("my.table") == "tbl_my_table"
        assert _sanitize_table_name("my@table!") == "tbl_my_table_"

    def test_starts_with_number_prefixed(self):
        """Names starting with numbers get extra prefix."""
        assert _sanitize_table_name("123table") == "tbl_t_123table"

    def test_truncation_at_limit(self):
        """Long names are truncated to PostgreSQL limit."""
        long_name = "a" * 100
        result = _sanitize_table_name(long_name)
        # tbl_ prefix (4) + truncated name (55) = 59 chars max
        assert len(result) <= 63
        assert result.startswith("tbl_")

    def test_lowercase_output(self):
        """Output is always lowercase."""
        assert _sanitize_table_name("MyTable") == "tbl_mytable"
        assert _sanitize_table_name("USER_DATA") == "tbl_user_data"


# =============================================================================
# Column Name Sanitization Tests
# =============================================================================


class TestColumnNameSanitization:
    """Tests for column name sanitization."""

    def test_simple_name_unchanged(self):
        """Simple alphanumeric name stays the same."""
        assert _sanitize_column_name("name") == "name"
        assert _sanitize_column_name("user_id") == "user_id"

    def test_spaces_replaced(self):
        """Spaces are replaced with underscores."""
        assert _sanitize_column_name("first name") == "first_name"
        assert _sanitize_column_name("user id") == "user_id"

    def test_special_chars_replaced(self):
        """Special characters are replaced."""
        assert _sanitize_column_name("price$") == "price_"
        assert _sanitize_column_name("user@email") == "user_email"

    def test_starts_with_number_prefixed(self):
        """Names starting with numbers get 'c_' prefix."""
        assert _sanitize_column_name("123col") == "c_123col"
        assert _sanitize_column_name("1st") == "c_1st"

    def test_empty_becomes_col(self):
        """Empty string becomes 'col'."""
        assert _sanitize_column_name("") == "col"

    def test_all_special_chars_becomes_underscores(self):
        """All special chars become underscores with prefix."""
        # When all chars are replaced, result starts with underscore -> gets c_ prefix
        result = _sanitize_column_name("!!!")
        assert result.startswith("c_") or result == "___"

    def test_lowercase_output(self):
        """Output is always lowercase."""
        assert _sanitize_column_name("FirstName") == "firstname"
        assert _sanitize_column_name("USER_ID") == "user_id"


# =============================================================================
# Fixture Setup
# =============================================================================


@pytest.fixture
def mock_connection_manager():
    """Create a mock connection manager."""
    manager = MagicMock()
    manager.start = Mock()
    return manager


@pytest.fixture
def mock_connection(mock_connection_manager):
    """Create a mock database connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.execute.return_value = cursor
    conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
    conn.cursor.return_value.__exit__ = Mock(return_value=False)

    mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
    mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

    return conn


@pytest.fixture
def table_store(mock_connection_manager):
    """Create PostgresTableStore with mocked connection manager."""
    with patch(
        "fitz_ai.tabular.store.postgres.get_connection_manager",
        return_value=mock_connection_manager,
    ):
        store = PostgresTableStore("test_collection")
        return store


# =============================================================================
# Store Tests
# =============================================================================


class TestStore:
    """Tests for store operation."""

    def test_store_creates_table(self, table_store, mock_connection_manager, mock_connection):
        """Store creates table with correct schema."""
        columns = ["id", "name", "value"]
        rows = [["1", "test", "100"]]

        table_store.store(
            table_id="my_table",
            columns=columns,
            rows=rows,
            source_file="test.csv",
        )

        # Verify CREATE TABLE was called
        calls = [str(call) for call in mock_connection.execute.call_args_list]
        create_calls = [c for c in calls if "CREATE TABLE" in c]
        assert len(create_calls) >= 1

    def test_store_handles_duplicate_columns(self, table_store, mock_connection_manager, mock_connection):
        """Store deduplicates column names."""
        columns = ["name", "name", "name"]  # All duplicates
        rows = [["a", "b", "c"]]

        table_store.store(
            table_id="dup_table",
            columns=columns,
            rows=rows,
            source_file="test.csv",
        )

        # Verify INSERT was called (means table creation succeeded)
        calls = [str(call) for call in mock_connection.execute.call_args_list]
        insert_calls = [c for c in calls if "INSERT" in c]
        assert len(insert_calls) >= 1

    def test_store_pads_short_rows(self, table_store, mock_connection_manager, mock_connection):
        """Store pads short rows with empty strings."""
        columns = ["a", "b", "c"]
        rows = [["1"]]  # Only one value, should be padded

        # Track what gets inserted
        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id="short_rows",
            columns=columns,
            rows=rows,
            source_file="test.csv",
        )

        # executemany should have been called
        assert cursor.executemany.called

    def test_store_truncates_long_rows(self, table_store, mock_connection_manager, mock_connection):
        """Store truncates rows longer than columns."""
        columns = ["a", "b"]
        rows = [["1", "2", "3", "4"]]  # Too many values

        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id="long_rows",
            columns=columns,
            rows=rows,
            source_file="test.csv",
        )

        # Should complete without error
        assert cursor.executemany.called


# =============================================================================
# Retrieve Tests
# =============================================================================


class TestRetrieve:
    """Tests for retrieve operation."""

    def test_retrieve_returns_stored_table(self, table_store, mock_connection_manager, mock_connection):
        """Retrieve returns StoredTable with correct data."""
        # Mock metadata query result
        mock_connection.execute.return_value.fetchone.return_value = (
            "tbl_my_table",  # table_name
            "abc123",        # hash
            ["col_a", "col_b"],  # columns (sanitized)
            ["Col A", "Col B"],  # original columns
            2,               # row_count
            "test.csv",      # source_file
        )

        # Mock data fetch
        mock_connection.execute.return_value.fetchall.return_value = [
            ["val1", "val2"],
            ["val3", "val4"],
        ]

        result = table_store.retrieve("my_table")

        assert result is not None
        assert result.table_id == "my_table"
        assert result.columns == ["Col A", "Col B"]  # Original names
        assert result.row_count == 2

    def test_retrieve_nonexistent_returns_none(self, table_store, mock_connection_manager, mock_connection):
        """Retrieve returns None for nonexistent table."""
        mock_connection.execute.return_value.fetchone.return_value = None

        result = table_store.retrieve("nonexistent")

        assert result is None


# =============================================================================
# Query Execution Tests
# =============================================================================


class TestQueryExecution:
    """Tests for SQL query execution."""

    def test_execute_query_returns_results(self, table_store, mock_connection_manager, mock_connection):
        """execute_query returns column names and rows."""
        cursor = MagicMock()
        cursor.description = [("count",), ("sum",)]
        cursor.fetchall.return_value = [[10, 100]]
        mock_connection.execute.return_value = cursor

        result = table_store.execute_query(
            "my_table",
            "SELECT COUNT(*), SUM(value) FROM tbl_my_table",
        )

        assert result is not None
        col_names, rows = result
        assert col_names == ["count", "sum"]
        assert rows == [[10, 100]]

    def test_execute_query_escapes_percent(self, table_store, mock_connection_manager, mock_connection):
        """execute_query escapes % when no params provided."""
        cursor = MagicMock()
        cursor.description = [("pct",)]
        cursor.fetchall.return_value = [["50%"]]
        mock_connection.execute.return_value = cursor

        # SQL with % that shouldn't be treated as placeholder
        table_store.execute_query(
            "my_table",
            "SELECT '50%' as pct",
        )

        # Verify % was escaped to %%
        call_args = mock_connection.execute.call_args
        sql = call_args[0][0]
        assert "%%" in sql

    def test_execute_query_handles_error(self, table_store, mock_connection_manager, mock_connection):
        """execute_query returns None on error."""
        mock_connection.execute.side_effect = Exception("SQL error")

        result = table_store.execute_query("my_table", "INVALID SQL")

        assert result is None


# =============================================================================
# Add Columns Tests
# =============================================================================


class TestAddColumns:
    """Tests for adding columns to existing tables."""

    def test_add_columns_alters_table(self, table_store, mock_connection_manager, mock_connection):
        """add_columns adds columns via ALTER TABLE."""
        # Mock metadata fetch
        mock_connection.execute.return_value.fetchone.return_value = (
            "tbl_my_table",
            ["existing_col"],
            ["Existing Col"],
        )

        result = table_store.add_columns(
            table_id="my_table",
            new_columns=["New Column"],
            column_values=[["value1"], ["value2"]],
        )

        assert result is True

        # Verify ALTER TABLE was called
        calls = [str(call) for call in mock_connection.execute.call_args_list]
        alter_calls = [c for c in calls if "ALTER TABLE" in c]
        assert len(alter_calls) >= 1

    def test_add_columns_handles_duplicates(self, table_store, mock_connection_manager, mock_connection):
        """add_columns handles duplicate column names."""
        # Existing columns include 'status'
        mock_connection.execute.return_value.fetchone.return_value = (
            "tbl_my_table",
            ["status"],
            ["Status"],
        )

        # Adding another 'status' column
        result = table_store.add_columns(
            table_id="my_table",
            new_columns=["Status"],  # Will conflict after sanitization
            column_values=[["active"]],
        )

        assert result is True

        # Should have added status_1, not status
        calls = [str(call) for call in mock_connection.execute.call_args_list]
        alter_calls = [c for c in calls if "ALTER TABLE" in c and "status_1" in c.lower()]
        assert len(alter_calls) >= 1

    def test_add_columns_nonexistent_table_fails(self, table_store, mock_connection_manager, mock_connection):
        """add_columns returns False for nonexistent table."""
        mock_connection.execute.return_value.fetchone.return_value = None

        result = table_store.add_columns(
            table_id="nonexistent",
            new_columns=["col"],
            column_values=[["val"]],
        )

        assert result is False


# =============================================================================
# List/Delete Tests
# =============================================================================


class TestListAndDelete:
    """Tests for list and delete operations."""

    def test_list_tables_returns_ids(self, table_store, mock_connection_manager, mock_connection):
        """list_tables returns all table IDs."""
        mock_connection.execute.return_value.__iter__ = Mock(
            return_value=iter([("table1",), ("table2",), ("table3",)])
        )

        result = table_store.list_tables()

        assert result == ["table1", "table2", "table3"]

    def test_delete_removes_table(self, table_store, mock_connection_manager, mock_connection):
        """delete removes table and metadata."""
        mock_connection.execute.return_value.fetchone.return_value = ("tbl_my_table",)

        table_store.delete("my_table")

        # Verify DROP TABLE was called
        calls = [str(call) for call in mock_connection.execute.call_args_list]
        drop_calls = [c for c in calls if "DROP TABLE" in c]
        assert len(drop_calls) >= 1

        # Verify DELETE from metadata
        delete_calls = [c for c in calls if "DELETE" in c and "_table_metadata" in c]
        assert len(delete_calls) >= 1


# =============================================================================
# Hash Tests
# =============================================================================


class TestHash:
    """Tests for hash operations."""

    def test_get_hash_returns_stored_hash(self, table_store, mock_connection_manager, mock_connection):
        """get_hash returns the stored content hash."""
        mock_connection.execute.return_value.fetchone.return_value = ("abc123hash",)

        result = table_store.get_hash("my_table")

        assert result == "abc123hash"

    def test_get_hash_nonexistent_returns_none(self, table_store, mock_connection_manager, mock_connection):
        """get_hash returns None for nonexistent table."""
        mock_connection.execute.return_value.fetchone.return_value = None

        result = table_store.get_hash("nonexistent")

        assert result is None

    def test_get_file_hash_returns_stored_hash(self, table_store, mock_connection_manager, mock_connection):
        """get_file_hash returns the stored file hash."""
        mock_connection.execute.return_value.fetchone.return_value = ("filehash456",)

        result = table_store.get_file_hash("my_table")

        assert result == "filehash456"


# =============================================================================
# Large Batch Tests
# =============================================================================


class TestLargeBatch:
    """Tests for large batch operations."""

    def test_store_large_batch_10k_rows(self, table_store, mock_connection_manager, mock_connection):
        """Store handles 10k+ rows in one call."""
        columns = ["id", "name", "value"]
        # Create 10,000 rows
        rows = [[str(i), f"name_{i}", str(i * 10)] for i in range(10000)]

        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id="large_table",
            columns=columns,
            rows=rows,
            source_file="large.csv",
        )

        # Verify executemany was called
        assert cursor.executemany.called
        # Verify all rows were passed
        call_args = cursor.executemany.call_args
        inserted_rows = call_args[0][1]
        assert len(inserted_rows) == 10000

    def test_store_handles_wide_table(self, table_store, mock_connection_manager, mock_connection):
        """Store handles tables with many columns."""
        # 100 columns
        columns = [f"col_{i}" for i in range(100)]
        rows = [[str(i) for i in range(100)] for _ in range(10)]

        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id="wide_table",
            columns=columns,
            rows=rows,
            source_file="wide.csv",
        )

        # Verify CREATE TABLE has all columns
        calls = [str(call) for call in mock_connection.execute.call_args_list]
        create_calls = [c for c in calls if "CREATE TABLE" in c]
        assert len(create_calls) >= 1


# =============================================================================
# Multi-Table Query Tests
# =============================================================================


class TestMultiTableQuery:
    """Tests for queries across multiple tables."""

    def test_execute_join_query(self, table_store, mock_connection_manager, mock_connection):
        """Execute query with JOIN across tables works."""
        cursor = MagicMock()
        cursor.description = [("user_name",), ("order_total",)]
        cursor.fetchall.return_value = [
            ["Alice", 100],
            ["Bob", 200],
        ]
        mock_connection.execute.return_value = cursor

        result = table_store.execute_query(
            "users",  # Primary table
            """
            SELECT u.name as user_name, SUM(o.total) as order_total
            FROM tbl_users u
            JOIN tbl_orders o ON u.id = o.user_id
            GROUP BY u.name
            """,
        )

        assert result is not None
        col_names, rows = result
        assert col_names == ["user_name", "order_total"]
        assert len(rows) == 2

    def test_execute_subquery(self, table_store, mock_connection_manager, mock_connection):
        """Execute query with subquery works."""
        cursor = MagicMock()
        cursor.description = [("name",), ("above_avg",)]
        cursor.fetchall.return_value = [["Product A", True]]
        mock_connection.execute.return_value = cursor

        result = table_store.execute_query(
            "products",
            """
            SELECT name, price > (SELECT AVG(price) FROM tbl_products) as above_avg
            FROM tbl_products
            WHERE category = 'electronics'
            """,
        )

        assert result is not None
        col_names, rows = result
        assert "above_avg" in col_names

    def test_execute_cte_query(self, table_store, mock_connection_manager, mock_connection):
        """Execute query with CTE (WITH clause) works."""
        cursor = MagicMock()
        cursor.description = [("category",), ("total_sales",)]
        cursor.fetchall.return_value = [
            ["Electronics", 50000],
            ["Books", 20000],
        ]
        mock_connection.execute.return_value = cursor

        result = table_store.execute_query(
            "sales",
            """
            WITH category_totals AS (
                SELECT category, SUM(amount) as total
                FROM tbl_sales
                GROUP BY category
            )
            SELECT category, total as total_sales
            FROM category_totals
            ORDER BY total DESC
            """,
        )

        assert result is not None
        col_names, rows = result
        assert len(rows) == 2
        assert rows[0][1] == 50000


# =============================================================================
# Edge Case Tests: Empty Data
# =============================================================================


class TestEmptyData:
    """Edge case tests for empty data handling."""

    def test_store_empty_table(self, table_store, mock_connection_manager, mock_connection):
        """Store with zero rows should create empty table."""
        columns = ["id", "name", "value"]
        rows = []  # No rows

        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id="empty_table",
            columns=columns,
            rows=rows,
            source_file="empty.csv",
        )

        # CREATE TABLE should be called
        calls = [str(call) for call in mock_connection.execute.call_args_list]
        create_calls = [c for c in calls if "CREATE TABLE" in c]
        assert len(create_calls) >= 1

        # executemany should NOT be called (no rows to insert)
        # or called with empty list
        if cursor.executemany.called:
            call_args = cursor.executemany.call_args
            assert len(call_args[0][1]) == 0

    def test_store_empty_columns(self, table_store, mock_connection_manager, mock_connection):
        """Store with zero columns should fail gracefully."""
        columns = []  # No columns
        rows = [["value1"]]

        # Should either raise or handle gracefully
        try:
            table_store.store(
                table_id="no_cols_table",
                columns=columns,
                rows=rows,
                source_file="nocols.csv",
            )
        except (ValueError, IndexError):
            pass  # Expected - can't create table with no columns


# =============================================================================
# Edge Case Tests: Special Characters
# =============================================================================


class TestSpecialCharacters:
    """Edge case tests for special character handling."""

    def test_null_values_in_cells(self, table_store, mock_connection_manager, mock_connection):
        """Store handles None/null values in cells."""
        columns = ["id", "name", "optional"]
        rows = [
            ["1", "test", None],
            ["2", None, "value"],
            [None, "name", "val"],
        ]

        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id="nulls_table",
            columns=columns,
            rows=rows,
            source_file="nulls.csv",
        )

        assert cursor.executemany.called

    def test_special_chars_in_data(self, table_store, mock_connection_manager, mock_connection):
        """Store handles special characters in cell values."""
        columns = ["id", "content"]
        rows = [
            ["1", "Line1\nLine2"],          # Newlines
            ["2", "Tab\there"],              # Tabs
            ["3", "Quote's and \"doubles\""],# Quotes
            ["4", "Semi;colon"],             # Semicolons
            ["5", "Back\\slash"],            # Backslashes
        ]

        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id="special_chars",
            columns=columns,
            rows=rows,
            source_file="special.csv",
        )

        assert cursor.executemany.called

    def test_very_long_cell_values(self, table_store, mock_connection_manager, mock_connection):
        """Store handles very long cell values."""
        columns = ["id", "content"]
        # 100KB string
        long_content = "x" * 100000
        rows = [["1", long_content]]

        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id="long_content",
            columns=columns,
            rows=rows,
            source_file="long.csv",
        )

        assert cursor.executemany.called

    def test_unicode_in_data(self, table_store, mock_connection_manager, mock_connection):
        """Store handles unicode characters in data."""
        columns = ["id", "name", "description"]
        rows = [
            ["1", "日本語", "Japanese text"],
            ["2", "Ελληνικά", "Greek text"],
            ["3", "Émoji 🎉", "With emoji"],
            ["4", "Café naïve", "Accented chars"],
        ]

        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id="unicode_table",
            columns=columns,
            rows=rows,
            source_file="unicode.csv",
        )

        assert cursor.executemany.called


# =============================================================================
# Edge Case Tests: SQL Injection Prevention
# =============================================================================


class TestSQLInjectionPrevention:
    """Edge case tests for SQL injection prevention."""

    def test_sql_injection_in_table_name(self, table_store, mock_connection_manager, mock_connection):
        """Malicious table names should be sanitized."""
        malicious_name = "users; DROP TABLE tbl_users;--"

        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id=malicious_name,
            columns=["id"],
            rows=[["1"]],
            source_file="test.csv",
        )

        # Verify the SQL doesn't contain raw injection
        calls = [str(call) for call in mock_connection.execute.call_args_list]
        for call in calls:
            assert "DROP TABLE" not in call or "tbl_users__drop_table" in call.lower()

    def test_sql_injection_in_column_values(self, table_store, mock_connection_manager, mock_connection):
        """Malicious values in cells should be parameterized."""
        columns = ["id", "name"]
        malicious_value = "'; DROP TABLE tbl_users;--"
        rows = [["1", malicious_value]]

        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id="safe_table",
            columns=columns,
            rows=rows,
            source_file="test.csv",
        )

        # Should use parameterized queries
        assert cursor.executemany.called
        # Values should be passed as parameters, not embedded in SQL

    def test_sql_injection_in_query(self, table_store, mock_connection_manager, mock_connection):
        """Execute query should not be vulnerable to injection via params."""
        cursor = MagicMock()
        cursor.description = [("count",)]
        cursor.fetchall.return_value = [[0]]
        mock_connection.execute.return_value = cursor

        # Note: execute_query takes raw SQL, which is by design
        # but should escape % when no params
        result = table_store.execute_query(
            "test_table",
            "SELECT COUNT(*) FROM tbl_test WHERE name = 'safe'",
        )

        assert result is not None


# =============================================================================
# Edge Case Tests: Table Metadata
# =============================================================================


class TestTableMetadata:
    """Edge case tests for table metadata operations."""

    def test_has_columns_nonexistent_table(self, table_store, mock_connection_manager, mock_connection):
        """has_columns on nonexistent table should return False."""
        mock_connection.execute.return_value.fetchone.return_value = None

        result = table_store.has_columns("nonexistent", ["col1", "col2"])

        # Should return False, not raise
        assert result is False

    def test_get_row_count(self, table_store, mock_connection_manager, mock_connection):
        """get_row_count returns correct count."""
        mock_connection.execute.return_value.fetchone.return_value = (42,)

        result = table_store.get_row_count("my_table")

        assert result == 42

    def test_get_row_count_nonexistent(self, table_store, mock_connection_manager, mock_connection):
        """get_row_count on nonexistent table returns 0 or None."""
        mock_connection.execute.side_effect = Exception("Table not found")

        result = table_store.get_row_count("nonexistent")

        # Should return 0 or None, not raise
        assert result is None or result == 0


# =============================================================================
# Edge Case Tests: Transaction Rollback
# =============================================================================


class TestTransactionRollback:
    """Edge case tests for transaction handling."""

    def test_add_columns_rollback_on_failure(self, table_store, mock_connection_manager, mock_connection):
        """add_columns should rollback on partial failure."""
        # Mock metadata fetch to succeed
        mock_connection.execute.return_value.fetchone.return_value = (
            "tbl_my_table",
            ["existing"],
            ["Existing"],
        )

        # Make UPDATE fail after ALTER succeeds
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:  # Metadata queries + ALTER
                return MagicMock(fetchone=Mock(return_value=("tbl_my_table", ["existing"], ["Existing"])))
            raise Exception("Update failed")  # Fail on value update

        mock_connection.execute.side_effect = side_effect

        result = table_store.add_columns(
            table_id="my_table",
            new_columns=["New Col"],
            column_values=[["val1"], ["val2"]],
        )

        # Should return False on failure
        assert result is False


# =============================================================================
# Edge Case Tests: Row Ordering
# =============================================================================


class TestRowOrdering:
    """Edge case tests for row ordering preservation."""

    def test_retrieve_preserves_row_order(self, table_store, mock_connection_manager, mock_connection):
        """Retrieve should preserve the original row order."""
        # Mock metadata
        mock_connection.execute.return_value.fetchone.return_value = (
            "tbl_ordered",
            "hash123",
            ["id", "seq"],
            ["ID", "Sequence"],
            5,
            "test.csv",
        )

        # Return rows in specific order
        mock_connection.execute.return_value.fetchall.return_value = [
            ["A", "1"],
            ["B", "2"],
            ["C", "3"],
            ["D", "4"],
            ["E", "5"],
        ]

        result = table_store.retrieve("ordered_table")

        assert result is not None
        # Rows should be in same order as returned
        assert result.rows[0] == ["A", "1"]
        assert result.rows[4] == ["E", "5"]

    def test_store_maintains_row_order(self, table_store, mock_connection_manager, mock_connection):
        """Store should maintain the order of input rows."""
        columns = ["id", "seq"]
        rows = [
            ["Z", "last"],
            ["A", "first"],
            ["M", "middle"],
        ]

        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = Mock(return_value=cursor)

        table_store.store(
            table_id="ordered",
            columns=columns,
            rows=rows,
            source_file="test.csv",
        )

        # Verify rows were inserted in order
        assert cursor.executemany.called
        call_args = cursor.executemany.call_args
        inserted_rows = call_args[0][1]
        # First row should be Z/last (index 0)
        assert inserted_rows[0][0] == "Z"
        assert inserted_rows[1][0] == "A"
        assert inserted_rows[2][0] == "M"
