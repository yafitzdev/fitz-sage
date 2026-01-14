# tests/tabular/test_store.py
"""Tests for table storage backends."""

import pytest

from fitz_ai.core.paths import FitzPaths
from fitz_ai.tabular.store.base import compress_csv, compute_hash, decompress_csv
from fitz_ai.tabular.store.cache import TableCache
from fitz_ai.tabular.store.sqlite import SqliteTableStore


class TestCompressionUtils:
    """Tests for compression utilities."""

    def test_compute_hash_deterministic(self):
        """Test that hash is deterministic."""
        columns = ["a", "b", "c"]
        rows = [["1", "2", "3"], ["4", "5", "6"]]

        hash1 = compute_hash(columns, rows)
        hash2 = compute_hash(columns, rows)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_compute_hash_different_content(self):
        """Test that different content produces different hash."""
        columns = ["a", "b"]
        rows1 = [["1", "2"]]
        rows2 = [["3", "4"]]

        hash1 = compute_hash(columns, rows1)
        hash2 = compute_hash(columns, rows2)

        assert hash1 != hash2

    def test_compress_decompress_roundtrip(self):
        """Test compression and decompression roundtrip."""
        columns = ["Name", "Age", "City"]
        rows = [
            ["Alice", "30", "New York"],
            ["Bob", "25", "Los Angeles"],
            ["Carol", "35", "Chicago"],
        ]

        compressed = compress_csv(columns, rows)
        decompressed_cols, decompressed_rows = decompress_csv(compressed)

        assert decompressed_cols == columns
        assert decompressed_rows == rows

    def test_compression_reduces_size(self):
        """Test that compression actually reduces size."""
        columns = ["A", "B", "C", "D", "E"]
        rows = [[f"value{i}{j}" for j in range(5)] for i in range(100)]

        # Calculate uncompressed size
        import csv
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(columns)
        writer.writerows(rows)
        uncompressed_size = len(buffer.getvalue().encode())

        compressed = compress_csv(columns, rows)

        # Compressed should be significantly smaller
        assert len(compressed) < uncompressed_size / 2

    def test_decompress_empty_table(self):
        """Test decompressing empty table."""
        columns = ["A", "B"]
        rows = []

        compressed = compress_csv(columns, rows)
        decompressed_cols, decompressed_rows = decompress_csv(compressed)

        assert decompressed_cols == columns
        assert decompressed_rows == []


class TestSqliteTableStore:
    """Tests for SQLite table store."""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create temporary workspace."""
        old_workspace = FitzPaths._workspace_override
        FitzPaths.set_workspace(tmp_path / ".fitz")
        yield tmp_path / ".fitz"
        FitzPaths._workspace_override = old_workspace

    @pytest.fixture
    def store(self, temp_workspace):
        """Create store with temp workspace."""
        store = SqliteTableStore("test_collection")
        yield store
        store.close()

    def test_store_and_retrieve(self, store):
        """Test basic store and retrieve."""
        columns = ["id", "name", "value"]
        rows = [
            ["1", "Alice", "100"],
            ["2", "Bob", "200"],
        ]

        hash = store.store("table1", columns, rows, "test.csv")

        result = store.retrieve("table1")

        assert result is not None
        assert result.table_id == "table1"
        assert result.columns == columns
        assert result.rows == rows
        assert result.hash == hash
        assert result.row_count == 2

    def test_retrieve_nonexistent(self, store):
        """Test retrieving nonexistent table."""
        result = store.retrieve("nonexistent")
        assert result is None

    def test_get_hash(self, store):
        """Test get_hash method."""
        columns = ["a", "b"]
        rows = [["1", "2"]]

        expected_hash = store.store("table2", columns, rows, "test.csv")

        result = store.get_hash("table2")
        assert result == expected_hash

        # Nonexistent table
        assert store.get_hash("nonexistent") is None

    def test_list_tables(self, store):
        """Test listing all tables."""
        store.store("table1", ["a"], [["1"]], "a.csv")
        store.store("table2", ["b"], [["2"]], "b.csv")
        store.store("table3", ["c"], [["3"]], "c.csv")

        tables = store.list_tables()

        assert set(tables) == {"table1", "table2", "table3"}

    def test_delete(self, store):
        """Test deleting a table."""
        store.store("table1", ["a"], [["1"]], "test.csv")

        assert store.retrieve("table1") is not None

        store.delete("table1")

        assert store.retrieve("table1") is None

    def test_update_existing(self, store):
        """Test updating existing table."""
        store.store("table1", ["a", "b"], [["1", "2"]], "test.csv")

        # Update with new data
        new_rows = [["3", "4"], ["5", "6"]]
        store.store("table1", ["a", "b"], new_rows, "test.csv")

        result = store.retrieve("table1")

        assert result.rows == new_rows
        assert result.row_count == 2


class TestTableCache:
    """Tests for table cache."""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create temporary workspace."""
        old_workspace = FitzPaths._workspace_override
        FitzPaths.set_workspace(tmp_path / ".fitz")
        yield tmp_path / ".fitz"
        FitzPaths._workspace_override = old_workspace

    @pytest.fixture
    def cache(self, temp_workspace):
        """Create cache with temp workspace."""
        cache = TableCache("test_collection")
        yield cache
        cache.close()

    def test_store_and_retrieve_with_matching_hash(self, cache):
        """Test retrieval with matching hash."""
        columns = ["a", "b"]
        rows = [["1", "2"]]
        hash = "abc123"

        cache.store("table1", hash, columns, rows)

        result = cache.retrieve("table1", expected_hash=hash)

        assert result is not None
        assert result.columns == columns
        assert result.rows == rows

    def test_retrieve_with_wrong_hash_returns_none(self, cache):
        """Test that wrong hash returns None (stale cache)."""
        columns = ["a", "b"]
        rows = [["1", "2"]]

        cache.store("table1", "old_hash", columns, rows)

        result = cache.retrieve("table1", expected_hash="new_hash")

        assert result is None

    def test_retrieve_nonexistent(self, cache):
        """Test retrieving nonexistent entry."""
        result = cache.retrieve("nonexistent", expected_hash="any")
        assert result is None

    def test_delete(self, cache):
        """Test deleting cache entry."""
        cache.store("table1", "hash", ["a"], [["1"]])

        cache.delete("table1")

        result = cache.retrieve("table1", expected_hash="hash")
        assert result is None

    def test_clear(self, cache):
        """Test clearing entire cache."""
        cache.store("table1", "h1", ["a"], [["1"]])
        cache.store("table2", "h2", ["b"], [["2"]])

        cache.clear()

        assert cache.retrieve("table1", expected_hash="h1") is None
        assert cache.retrieve("table2", expected_hash="h2") is None
