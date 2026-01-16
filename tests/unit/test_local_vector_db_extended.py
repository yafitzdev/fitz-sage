# tests/test_local_vector_db_extended.py
"""
Extended tests for local vector database backends.

Tests additional functionality not covered in test_local_faiss_vector_db.py:
- scroll and scroll_with_vectors
- list_collections
- get_collection_stats
- config
- runtime
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("faiss")


class TestLocalVectorDBConfig:
    """Tests for LocalVectorDBConfig."""

    def test_default_path_uses_fitz_paths(self, tmp_path):
        """Test that default path comes from FitzPaths."""
        with patch(
            "fitz_ai.core.paths.FitzPaths.vector_db",
            return_value=tmp_path / "vector_db",
        ):
            from fitz_ai.backends.local_vector_db.config import LocalVectorDBConfig

            cfg = LocalVectorDBConfig()

            assert cfg.path == tmp_path / "vector_db"

    def test_custom_path(self, tmp_path):
        """Test custom path configuration."""
        from fitz_ai.backends.local_vector_db.config import LocalVectorDBConfig

        custom_path = tmp_path / "custom" / "vectors"
        cfg = LocalVectorDBConfig(path=custom_path)

        assert cfg.path == custom_path

    def test_default_persist_true(self, tmp_path):
        """Test that persist defaults to True."""
        with patch("fitz_ai.core.paths.FitzPaths.vector_db", return_value=tmp_path):
            from fitz_ai.backends.local_vector_db.config import LocalVectorDBConfig

            cfg = LocalVectorDBConfig()

            assert cfg.persist is True

    def test_persist_can_be_disabled(self, tmp_path):
        """Test that persist can be set to False."""
        from fitz_ai.backends.local_vector_db.config import LocalVectorDBConfig

        cfg = LocalVectorDBConfig(path=tmp_path, persist=False)

        assert cfg.persist is False


class TestFaissListCollections:
    """Tests for list_collections method."""

    def test_list_collections_empty(self, tmp_path: Path):
        """Test list_collections on empty database."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        result = db.list_collections()

        assert result == []

    def test_list_collections_single(self, tmp_path: Path):
        """Test list_collections with single collection."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        db.upsert("my_collection", [{"id": "1", "vector": [0.1, 0.2], "payload": {}}])

        result = db.list_collections()

        assert result == ["my_collection"]

    def test_list_collections_multiple(self, tmp_path: Path):
        """Test list_collections with multiple collections."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        db.upsert("zebra", [{"id": "1", "vector": [0.1, 0.2], "payload": {}}])
        db.upsert("apple", [{"id": "2", "vector": [0.3, 0.4], "payload": {}}])
        db.upsert("middle", [{"id": "3", "vector": [0.5, 0.6], "payload": {}}])

        result = db.list_collections()

        # Should be sorted alphabetically
        assert result == ["apple", "middle", "zebra"]

    def test_list_collections_no_duplicates(self, tmp_path: Path):
        """Test list_collections returns unique names."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        db.upsert("same", [{"id": "1", "vector": [0.1, 0.2], "payload": {}}])
        db.upsert("same", [{"id": "2", "vector": [0.3, 0.4], "payload": {}}])
        db.upsert("same", [{"id": "3", "vector": [0.5, 0.6], "payload": {}}])

        result = db.list_collections()

        assert result == ["same"]


class TestFaissGetCollectionStats:
    """Tests for get_collection_stats method."""

    def test_stats_empty_collection(self, tmp_path: Path):
        """Test stats for non-existent collection."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        db.upsert("other", [{"id": "1", "vector": [0.1, 0.2], "payload": {}}])

        stats = db.get_collection_stats("nonexistent")

        assert stats["points_count"] == 0
        assert stats["vectors_count"] == 0
        assert stats["status"] == "not_found"  # Non-existent collection

    def test_stats_with_data(self, tmp_path: Path):
        """Test stats for collection with data."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        db.upsert(
            "docs",
            [
                {"id": "1", "vector": [0.1, 0.2, 0.3, 0.4], "payload": {}},
                {"id": "2", "vector": [0.5, 0.6, 0.7, 0.8], "payload": {}},
                {"id": "3", "vector": [0.9, 1.0, 1.1, 1.2], "payload": {}},
            ],
        )

        stats = db.get_collection_stats("docs")

        assert stats["points_count"] == 3
        assert stats["vectors_count"] == 3
        assert stats["vector_size"] == 4
        assert stats["status"] == "ready"


class TestFaissScroll:
    """Tests for scroll method."""

    def test_scroll_empty(self, tmp_path: Path):
        """Test scroll on empty database."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        records, next_offset = db.scroll("collection", limit=10)

        assert records == []
        assert next_offset is None

    def test_scroll_basic(self, tmp_path: Path):
        """Test basic scroll functionality."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        db.upsert(
            "docs",
            [
                {"id": "d1", "vector": [0.1, 0.2], "payload": {"content": "Doc 1"}},
                {"id": "d2", "vector": [0.3, 0.4], "payload": {"content": "Doc 2"}},
            ],
        )

        records, next_offset = db.scroll("docs", limit=10)

        assert len(records) == 2
        assert records[0].id == "d1"
        assert records[0].payload["content"] == "Doc 1"
        assert next_offset is None  # All records returned

    def test_scroll_with_offset(self, tmp_path: Path):
        """Test scroll with offset."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        db.upsert(
            "docs",
            [{"id": f"d{i}", "vector": [0.1 * i, 0.2 * i], "payload": {"i": i}} for i in range(5)],
        )

        records, next_offset = db.scroll("docs", limit=2, offset=2)

        assert len(records) == 2
        assert records[0].id == "d2"
        assert records[1].id == "d3"
        assert next_offset == 4  # More records available

    def test_scroll_pagination(self, tmp_path: Path):
        """Test scroll pagination through all records."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        db.upsert(
            "docs",
            [{"id": f"d{i}", "vector": [0.1, 0.2], "payload": {}} for i in range(5)],
        )

        all_ids = []
        offset = 0

        while True:
            records, next_offset = db.scroll("docs", limit=2, offset=offset)
            all_ids.extend([r.id for r in records])

            if next_offset is None:
                break
            offset = next_offset

        assert len(all_ids) == 5
        assert set(all_ids) == {f"d{i}" for i in range(5)}

    def test_scroll_filters_by_collection(self, tmp_path: Path):
        """Test scroll only returns records from specified collection."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        db.upsert("coll_a", [{"id": "a1", "vector": [0.1, 0.2], "payload": {}}])
        db.upsert("coll_b", [{"id": "b1", "vector": [0.3, 0.4], "payload": {}}])

        records_a, _ = db.scroll("coll_a", limit=10)
        records_b, _ = db.scroll("coll_b", limit=10)

        assert len(records_a) == 1
        assert records_a[0].id == "a1"
        assert len(records_b) == 1
        assert records_b[0].id == "b1"

    def test_scroll_removes_internal_fields(self, tmp_path: Path):
        """Test scroll removes _collection from payload."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        db.upsert("docs", [{"id": "d1", "vector": [0.1, 0.2], "payload": {"key": "value"}}])

        records, _ = db.scroll("docs", limit=10)

        assert "_collection" not in records[0].payload
        assert records[0].payload["key"] == "value"


class TestFaissScrollWithVectors:
    """Tests for scroll_with_vectors method."""

    def test_scroll_with_vectors_empty(self, tmp_path: Path):
        """Test scroll_with_vectors on empty database."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        records, next_offset = db.scroll_with_vectors("collection", limit=10)

        assert records == []
        assert next_offset is None

    def test_scroll_with_vectors_includes_vector(self, tmp_path: Path):
        """Test scroll_with_vectors includes vector in results."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        original_vector = [0.1, 0.2, 0.3, 0.4]
        db.upsert(
            "docs",
            [{"id": "d1", "vector": original_vector, "payload": {"content": "Test"}}],
        )

        records, _ = db.scroll_with_vectors("docs", limit=10)

        assert len(records) == 1
        assert records[0]["id"] == "d1"
        assert records[0]["payload"]["content"] == "Test"
        assert "vector" in records[0]
        # Vector should match (approximately due to float conversion)
        assert len(records[0]["vector"]) == 4
        for i, v in enumerate(records[0]["vector"]):
            assert abs(v - original_vector[i]) < 0.001

    def test_scroll_with_vectors_pagination(self, tmp_path: Path):
        """Test scroll_with_vectors pagination."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)
        db.upsert(
            "docs",
            [{"id": f"d{i}", "vector": [0.1 * i, 0.2], "payload": {}} for i in range(5)],
        )

        records, next_offset = db.scroll_with_vectors("docs", limit=2, offset=0)

        assert len(records) == 2
        assert next_offset == 2

        records2, next_offset2 = db.scroll_with_vectors("docs", limit=2, offset=2)

        assert len(records2) == 2
        assert next_offset2 == 4


class TestFaissFlush:
    """Tests for flush method."""

    def test_flush_persists_data(self, tmp_path: Path):
        """Test flush persists index to disk."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path, persist=True)
        db.upsert(
            "docs",
            [{"id": "1", "vector": [0.1, 0.2], "payload": {}}],
            defer_persist=True,
        )

        # Before flush, file might not exist or be stale
        db.flush()

        # After flush, should be able to reload
        db2 = FaissLocalVectorDB(path=tmp_path, persist=True)
        assert db2.count() == 1

    def test_flush_no_op_when_not_persist(self, tmp_path: Path):
        """Test flush does nothing when persist=False."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path, persist=False)
        db.upsert("docs", [{"id": "1", "vector": [0.1, 0.2], "payload": {}}])

        db.flush()  # Should not raise

        # No files should be created (per-collection storage)
        assert not (tmp_path / "docs" / "index.faiss").exists()


class TestFaissDeferPersist:
    """Tests for defer_persist option."""

    def test_upsert_with_defer_persist(self, tmp_path: Path):
        """Test upsert with defer_persist=True doesn't auto-save."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path, persist=True)

        # First upsert without defer
        db.upsert("docs", [{"id": "1", "vector": [0.1, 0.2], "payload": {}}])

        # Record when file was last modified (per-collection storage)
        import time

        collection_index = tmp_path / "docs" / "index.faiss"
        time.sleep(0.01)
        mtime1 = collection_index.stat().st_mtime

        # Upsert with defer - should NOT update file
        time.sleep(0.01)
        db.upsert(
            "docs",
            [{"id": "2", "vector": [0.3, 0.4], "payload": {}}],
            defer_persist=True,
        )

        mtime2 = collection_index.stat().st_mtime

        # File modification time should be unchanged
        assert mtime1 == mtime2

        # But data should be in memory
        assert db.count() == 2


class TestFaissPluginAttributes:
    """Tests for plugin metadata attributes."""

    def test_plugin_name(self, tmp_path: Path):
        """Test plugin_name attribute."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)

        assert db.plugin_name == "local-faiss"

    def test_plugin_type(self, tmp_path: Path):
        """Test plugin_type attribute."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=tmp_path)

        assert db.plugin_type == "vector_db"
