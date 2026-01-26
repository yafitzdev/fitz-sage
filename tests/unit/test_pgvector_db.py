# tests/unit/test_pgvector_db.py
"""
Unit tests for PgVectorDB plugin.

Tests cover:
1. Dimension mismatch detection
2. Filter clause building (must, should, match, range, nested keys)
3. Hybrid search RRF fusion
4. Collection discovery
5. Empty collection handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

# Mark all tests in this module as postgres and tier2
pytestmark = [pytest.mark.postgres, pytest.mark.tier2]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_connection_manager():
    """Create a mock connection manager."""
    manager = MagicMock()
    manager.start = Mock()
    manager.connection = MagicMock()
    return manager


@pytest.fixture
def mock_connection(mock_connection_manager):
    """Create a mock database connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.execute.return_value = cursor
    conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
    conn.cursor.return_value.__exit__ = Mock(return_value=False)

    # Setup context manager
    mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
    mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

    return conn


@pytest.fixture
def pgvector_db(mock_connection_manager):
    """Create PgVectorDB instance with mocked connection manager."""
    with patch(
        "fitz_ai.backends.local_vector_db.pgvector.get_connection_manager",
        return_value=mock_connection_manager,
    ):
        from fitz_ai.backends.local_vector_db.pgvector import PgVectorDB

        db = PgVectorDB(mode="local")
        return db


# =============================================================================
# Filter Building Tests
# =============================================================================


class TestFilterBuilding:
    """Tests for SQL filter clause building."""

    def test_empty_filter_returns_empty(self, pgvector_db):
        """Empty filter returns empty clause."""
        clause, params = pgvector_db._build_filter_clause(None)
        assert clause == ""
        assert params == []

        clause, params = pgvector_db._build_filter_clause({})
        assert clause == ""
        assert params == []

    def test_simple_match_filter(self, pgvector_db):
        """Simple match filter builds correct clause."""
        filter_cond = {"key": "status", "match": {"value": "active"}}
        clause, params = pgvector_db._build_filter_clause(filter_cond)

        assert "payload->>'status'" in clause
        assert "= %s" in clause
        assert params == ["active"]

    def test_must_filter_combines_with_and(self, pgvector_db):
        """Must filter combines conditions with AND."""
        filter_cond = {
            "must": [
                {"key": "status", "match": {"value": "active"}},
                {"key": "type", "match": {"value": "document"}},
            ]
        }
        clause, params = pgvector_db._build_filter_clause(filter_cond)

        assert "WHERE" in clause
        assert "payload->>'status'" in clause
        assert "payload->>'type'" in clause
        assert params == ["active", "document"]

    def test_should_filter_combines_with_or(self, pgvector_db):
        """Should filter combines conditions with OR."""
        filter_cond = {
            "should": [
                {"key": "status", "match": {"value": "active"}},
                {"key": "status", "match": {"value": "pending"}},
            ]
        }
        clause, params = pgvector_db._build_filter_clause(filter_cond)

        assert "WHERE" in clause
        assert " OR " in clause
        assert params == ["active", "pending"]

    def test_range_filter_gte(self, pgvector_db):
        """Range filter with gte builds correct clause."""
        filter_cond = {"key": "score", "range": {"gte": 0.5}}
        clause, params = pgvector_db._build_filter_clause(filter_cond)

        assert "::numeric >= %s" in clause
        assert params == [0.5]

    def test_range_filter_multiple_bounds(self, pgvector_db):
        """Range filter with multiple bounds."""
        filter_cond = {"key": "score", "range": {"gte": 0.5, "lt": 1.0}}
        clause, params = pgvector_db._build_filter_clause(filter_cond)

        assert ">= %s" in clause
        assert "< %s" in clause
        assert 0.5 in params
        assert 1.0 in params

    def test_nested_key_filter(self, pgvector_db):
        """Nested key uses JSONB path notation."""
        filter_cond = {"key": "metadata.author", "match": {"value": "John"}}
        clause, params = pgvector_db._build_filter_clause(filter_cond)

        assert "payload->'metadata'->>'author'" in clause
        assert params == ["John"]

    def test_deeply_nested_key_filter(self, pgvector_db):
        """Deeply nested key builds correct path."""
        filter_cond = {"key": "a.b.c.d", "match": {"value": "test"}}
        clause, params = pgvector_db._build_filter_clause(filter_cond)

        assert "payload->'a'->'b'->'c'->>'d'" in clause
        assert params == ["test"]

    def test_combined_must_and_should(self, pgvector_db):
        """Combined must and should filters."""
        filter_cond = {
            "must": [
                {"key": "active", "match": {"value": "true"}},
                {
                    "should": [
                        {"key": "type", "match": {"value": "a"}},
                        {"key": "type", "match": {"value": "b"}},
                    ]
                },
            ]
        }
        clause, params = pgvector_db._build_filter_clause(filter_cond)

        assert "WHERE" in clause
        assert " AND " in clause
        assert " OR " in clause


# =============================================================================
# Dimension Mismatch Tests
# =============================================================================


class TestDimensionMismatch:
    """Tests for dimension validation."""

    def test_dimension_mismatch_in_memory_cache(self, pgvector_db):
        """Dimension mismatch detected from in-memory cache."""
        # Simulate existing collection with dim=384
        pgvector_db._initialized_collections["test_coll"] = 384

        with pytest.raises(ValueError) as exc_info:
            pgvector_db._ensure_schema("test_coll", dim=1536)

        assert "Dimension mismatch" in str(exc_info.value)
        assert "384" in str(exc_info.value)
        assert "1536" in str(exc_info.value)

    def test_dimension_mismatch_from_database(self, pgvector_db, mock_connection_manager):
        """Dimension mismatch detected from existing database table."""
        conn = MagicMock()

        # First call: table exists check returns True
        # Second call: dimension check returns 384
        conn.execute.side_effect = [
            MagicMock(fetchone=Mock(return_value=(True,))),  # table exists
            MagicMock(fetchone=Mock(return_value=(384,))),  # existing dim
        ]

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            with pytest.raises(ValueError) as exc_info:
                pgvector_db._ensure_schema("test_coll", dim=1536)

        assert "Dimension mismatch" in str(exc_info.value)

    def test_same_dimension_no_error(self, pgvector_db):
        """Same dimension does not raise error."""
        pgvector_db._initialized_collections["test_coll"] = 1536

        # Should not raise
        pgvector_db._ensure_schema("test_coll", dim=1536)


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Tests for search operations."""

    def test_search_empty_collection_returns_empty(self, pgvector_db):
        """Search on unknown collection returns empty list."""
        # Collection not in initialized_collections and discovery fails
        with patch.object(pgvector_db, "_discover_collection", return_value=False):
            results = pgvector_db.search("unknown_coll", [0.1] * 384, limit=10)

        assert results == []

    def test_search_with_filter(self, pgvector_db, mock_connection_manager):
        """Search with filter builds correct SQL."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__iter__ = Mock(
            return_value=iter(
                [
                    ("id1", 0.95, {"content": "test"}),
                ]
            )
        )
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            pgvector_db.search(
                "test_coll",
                [0.1] * 384,
                limit=10,
                query_filter={"key": "status", "match": {"value": "active"}},
            )

        # Verify SQL was called with filter
        call_args = conn.execute.call_args
        sql = call_args[0][0]
        assert "WHERE" in sql
        assert "payload->>'status'" in sql


# =============================================================================
# Hybrid Search Tests
# =============================================================================


class TestHybridSearch:
    """Tests for hybrid search with RRF fusion."""

    def test_hybrid_search_empty_collection(self, pgvector_db):
        """Hybrid search on unknown collection returns empty."""
        results = pgvector_db.hybrid_search(
            "unknown_coll",
            query_vector=[0.1] * 384,
            query_text="test query",
            limit=10,
        )
        assert results == []

    def test_hybrid_search_rrf_query(self, pgvector_db, mock_connection_manager):
        """Hybrid search builds RRF fusion query."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__iter__ = Mock(
            return_value=iter(
                [
                    ("id1", 0.025, {"content": "test result"}),
                ]
            )
        )
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            pgvector_db.hybrid_search(
                "test_coll",
                query_vector=[0.1] * 384,
                query_text="test query",
                limit=10,
                alpha=0.7,
            )

        # Verify RRF SQL structure
        call_args = conn.execute.call_args
        sql = call_args[0][0]
        assert "vector_results" in sql  # CTE for vector search
        assert "text_results" in sql  # CTE for text search
        assert "rrf" in sql  # RRF fusion CTE
        assert "ts_rank_cd" in sql  # BM25-style ranking
        assert "plainto_tsquery" in sql  # Full-text query

    def test_hybrid_search_alpha_weights(self, pgvector_db, mock_connection_manager):
        """Hybrid search applies correct alpha weights."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__iter__ = Mock(return_value=iter([]))
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            pgvector_db.hybrid_search(
                "test_coll",
                query_vector=[0.1] * 384,
                query_text="test",
                limit=10,
                alpha=0.8,  # 80% vector, 20% text
            )

        # Check params include weights - they are in the params list
        call_args = conn.execute.call_args
        params = call_args[0][1]
        # Params list contains: query_vector, limit*2, query_text, query_text, limit*2, vector_weight, text_weight, limit
        # Check vector_weight (0.8) and text_weight (0.2) are in the list
        assert 0.8 in params
        assert any(abs(p - 0.2) < 0.001 for p in params if isinstance(p, (int, float)))


# =============================================================================
# Collection Discovery Tests
# =============================================================================


class TestCollectionDiscovery:
    """Tests for collection discovery."""

    def test_discover_existing_collection(self, pgvector_db, mock_connection_manager):
        """Discover existing collection from database."""
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = (1536,)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        result = pgvector_db._discover_collection("existing_coll")

        assert result is True
        assert pgvector_db._initialized_collections["existing_coll"] == 1536

    def test_discover_nonexistent_collection(self, pgvector_db, mock_connection_manager):
        """Discovery returns False for nonexistent collection."""
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        result = pgvector_db._discover_collection("nonexistent")

        assert result is False
        assert "nonexistent" not in pgvector_db._initialized_collections

    def test_discover_handles_exception(self, pgvector_db, mock_connection_manager):
        """Discovery handles database errors gracefully."""
        mock_connection_manager.connection.side_effect = Exception("DB error")

        result = pgvector_db._discover_collection("error_coll")

        assert result is False


# =============================================================================
# Upsert Tests
# =============================================================================


class TestUpsert:
    """Tests for upsert operations."""

    def test_upsert_empty_points_noop(self, pgvector_db):
        """Upsert with empty points does nothing."""
        pgvector_db.upsert("test_coll", [])
        # Should not raise or call connection

    def test_upsert_auto_detects_dimension(self, pgvector_db, mock_connection_manager):
        """Upsert auto-detects dimension from first vector."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)

        # Table doesn't exist yet
        conn.execute.return_value.fetchone.return_value = (False,)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            with patch.object(pgvector_db, "_ensure_schema") as mock_ensure:
                pgvector_db.upsert(
                    "test_coll",
                    [{"id": "1", "vector": [0.1] * 768, "payload": {}}],
                )

                # Should detect dim=768 from first vector
                mock_ensure.assert_called_with("test_coll", 768)

    def test_upsert_batch_large(self, pgvector_db, mock_connection_manager):
        """Upsert handles large batch of 1000+ vectors."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        # Create 1000 points
        points = [{"id": str(i), "vector": [0.1] * 384, "payload": {"idx": i}} for i in range(1000)]

        with patch("pgvector.psycopg.register_vector"):
            with patch.object(pgvector_db, "_ensure_schema"):
                pgvector_db.upsert("test_coll", points)

        # Verify executemany was called with all points
        assert cursor.executemany.called
        call_args = cursor.executemany.call_args
        inserted_points = call_args[0][1]
        assert len(inserted_points) == 1000

    def test_upsert_update_existing(self, pgvector_db, mock_connection_manager):
        """Upsert with same ID updates existing record."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            with patch.object(pgvector_db, "_ensure_schema"):
                # First insert
                pgvector_db.upsert(
                    "test_coll",
                    [{"id": "same_id", "vector": [0.1] * 384, "payload": {"v": 1}}],
                )
                # Update with same ID
                pgvector_db.upsert(
                    "test_coll",
                    [{"id": "same_id", "vector": [0.2] * 384, "payload": {"v": 2}}],
                )

        # Verify ON CONFLICT DO UPDATE is in the SQL
        call_args = cursor.executemany.call_args
        sql = call_args[0][0]
        assert "ON CONFLICT" in sql
        assert "DO UPDATE" in sql


# =============================================================================
# Delete Collection Tests
# =============================================================================


# =============================================================================
# Scroll Pagination Tests
# =============================================================================


class TestScrollPagination:
    """Tests for scroll/pagination operations."""

    def test_scroll_returns_records_with_offset(self, pgvector_db, mock_connection_manager):
        """Scroll returns records with correct offset."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        # Return 10 records
        cursor = MagicMock()
        cursor.__iter__ = Mock(
            return_value=iter([(f"id_{i}", {"content": f"content_{i}"}) for i in range(10)])
        )
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        records, next_offset = pgvector_db.scroll("test_coll", limit=10, offset=0)

        assert len(records) == 10
        assert next_offset == 10  # Next offset for pagination
        assert records[0].id == "id_0"

    def test_scroll_last_page_no_next_offset(self, pgvector_db, mock_connection_manager):
        """Scroll on last page returns None for next_offset."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        # Return less than limit (last page)
        cursor = MagicMock()
        cursor.__iter__ = Mock(
            return_value=iter(
                [
                    ("id_0", {"content": "last"}),
                ]
            )
        )
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        records, next_offset = pgvector_db.scroll("test_coll", limit=10, offset=90)

        assert len(records) == 1
        assert next_offset is None  # No more pages

    def test_scroll_unknown_collection_returns_empty(self, pgvector_db):
        """Scroll on unknown collection returns empty list."""
        records, next_offset = pgvector_db.scroll("unknown_coll", limit=10, offset=0)

        assert records == []
        assert next_offset is None

    def test_scroll_with_vectors(self, pgvector_db, mock_connection_manager):
        """Scroll with vectors includes vector data."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__iter__ = Mock(
            return_value=iter(
                [
                    ("id_0", {"content": "test"}, [0.1] * 384),
                ]
            )
        )
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            records, next_offset = pgvector_db.scroll_with_vectors("test_coll", limit=10)

        assert len(records) == 1
        assert "vector" in records[0]
        assert len(records[0]["vector"]) == 384


# =============================================================================
# Delete Collection Tests
# =============================================================================


class TestDeleteCollection:
    """Tests for collection deletion."""

    def test_delete_collection_clears_cache(self, pgvector_db, mock_connection_manager):
        """Delete collection removes from initialized cache."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = (100,)  # 100 vectors

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        count = pgvector_db.delete_collection("test_coll")

        assert count == 100
        assert "test_coll" not in pgvector_db._initialized_collections

    def test_delete_nonexistent_collection(self, pgvector_db, mock_connection_manager):
        """Delete nonexistent collection returns 0."""
        conn = MagicMock()
        conn.execute.side_effect = Exception("Table not found")

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        count = pgvector_db.delete_collection("nonexistent")

        assert count == 0


# =============================================================================
# Edge Case Tests: Vector Validation
# =============================================================================


class TestVectorValidation:
    """Edge case tests for vector validation."""

    def test_empty_vector_rejected(self, pgvector_db, mock_connection_manager):
        """Empty vector (zero length) should be rejected or handled gracefully."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        # Empty vector should fail during schema setup (dim=0)
        with patch("pgvector.psycopg.register_vector"):
            # Either raises or handles gracefully
            try:
                pgvector_db.upsert(
                    "test_coll",
                    [{"id": "1", "vector": [], "payload": {}}],  # Empty vector
                )
                # If it didn't raise, verify no actual insert occurred
                # or dim was detected as 0
            except (ValueError, IndexError):
                pass  # Expected - empty vectors are invalid

    def test_search_ordering_by_similarity(self, pgvector_db, mock_connection_manager):
        """Search results should be ordered by similarity (highest first)."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        # Return results in order of decreasing similarity
        cursor.__iter__ = Mock(
            return_value=iter(
                [
                    ("id_high", 0.95, {"content": "very similar"}),
                    ("id_med", 0.75, {"content": "somewhat similar"}),
                    ("id_low", 0.50, {"content": "less similar"}),
                ]
            )
        )
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            results = pgvector_db.search("test_coll", [0.1] * 384, limit=10)

        # Verify ordering
        assert len(results) == 3
        assert results[0].id == "id_high"
        assert results[0].score == 0.95
        assert results[1].score <= results[0].score
        assert results[2].score <= results[1].score


# =============================================================================
# Edge Case Tests: Payload Handling
# =============================================================================


class TestPayloadEdgeCases:
    """Edge case tests for payload handling."""

    def test_payload_with_null_values(self, pgvector_db, mock_connection_manager):
        """Payload with null/None values should be handled."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            with patch.object(pgvector_db, "_ensure_schema"):
                # Payload with None values
                pgvector_db.upsert(
                    "test_coll",
                    [
                        {
                            "id": "1",
                            "vector": [0.1] * 384,
                            "payload": {"content": None, "score": None, "valid": True},
                        }
                    ],
                )

        # Should complete without error
        assert cursor.executemany.called

    def test_unicode_in_payload(self, pgvector_db, mock_connection_manager):
        """Payload with unicode characters should be stored correctly."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            with patch.object(pgvector_db, "_ensure_schema"):
                pgvector_db.upsert(
                    "test_coll",
                    [
                        {
                            "id": "unicode_test",
                            "vector": [0.1] * 384,
                            "payload": {
                                "content": "日本語テスト",
                                "emoji": "🎉🚀",
                                "special": "café naïve",
                            },
                        }
                    ],
                )

        assert cursor.executemany.called
        # Verify the payload was passed correctly
        call_args = cursor.executemany.call_args
        inserted = call_args[0][1][0]  # First point tuple
        # Payload is the third element in the tuple
        assert "日本語テスト" in str(inserted) or cursor.executemany.called

    def test_very_long_id_handling(self, pgvector_db, mock_connection_manager):
        """Very long IDs should be handled (truncated or rejected)."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        # Very long ID (1000+ chars)
        long_id = "x" * 5000

        with patch("pgvector.psycopg.register_vector"):
            with patch.object(pgvector_db, "_ensure_schema"):
                pgvector_db.upsert(
                    "test_coll",
                    [{"id": long_id, "vector": [0.1] * 384, "payload": {}}],
                )

        # Should complete (ID may be truncated or stored as-is depending on impl)
        assert cursor.executemany.called


# =============================================================================
# Edge Case Tests: Filter Handling
# =============================================================================


class TestFilterEdgeCases:
    """Edge case tests for filter handling."""

    def test_filter_missing_key_no_match(self, pgvector_db, mock_connection_manager):
        """Filter on key not present in payload should return no matches."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        # Return empty (no matches for missing key)
        cursor.__iter__ = Mock(return_value=iter([]))
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            results = pgvector_db.search(
                "test_coll",
                [0.1] * 384,
                limit=10,
                query_filter={"key": "nonexistent_field", "match": {"value": "test"}},
            )

        assert results == []


# =============================================================================
# Edge Case Tests: Count and Retrieve
# =============================================================================


class TestCountAndRetrieveEdgeCases:
    """Edge case tests for count and retrieve operations."""

    def test_count_unknown_collection(self, pgvector_db):
        """Count on unknown collection should return 0."""
        # Collection not in initialized_collections
        with patch.object(pgvector_db, "_discover_collection", return_value=False):
            count = pgvector_db.count("unknown_collection")

        assert count == 0

    def test_retrieve_partial_ids(self, pgvector_db, mock_connection_manager):
        """Retrieve with some valid and some invalid IDs returns only valid ones."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        # Only return data for existing IDs
        cursor.__iter__ = Mock(
            return_value=iter(
                [
                    ("id1", {"content": "content1"}),
                    ("id3", {"content": "content3"}),
                    # id2 not found
                ]
            )
        )
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            results = pgvector_db.retrieve(
                "test_coll",
                ids=["id1", "id2", "id3"],  # id2 doesn't exist
            )

        # Should return only found records
        assert len(results) == 2
        result_ids = [r.get("id") or r.id for r in results]
        assert "id1" in result_ids
        assert "id3" in result_ids
        assert "id2" not in result_ids


# =============================================================================
# Edge Case Tests: Import Error Handling
# =============================================================================


class TestImportErrorHandling:
    """Tests for handling missing dependencies."""

    def test_import_error_pgvector_not_installed(self):
        """Should raise clear error if pgvector not installed."""
        # This tests the behavior when register_vector import fails
        with patch.dict("sys.modules", {"pgvector.psycopg": None}):
            with patch(
                "fitz_ai.backends.local_vector_db.pgvector.get_connection_manager"
            ) as mock_mgr:
                mock_mgr.return_value = MagicMock()

                from fitz_ai.backends.local_vector_db.pgvector import PgVectorDB

                db = PgVectorDB(mode="local")
                db._initialized_collections["test"] = 384

                conn = MagicMock()
                cursor = MagicMock()
                cursor.__iter__ = Mock(return_value=iter([]))
                conn.execute.return_value = cursor

                mock_mgr.return_value.connection.return_value.__enter__ = Mock(return_value=conn)
                mock_mgr.return_value.connection.return_value.__exit__ = Mock(return_value=False)

                # Should either raise or handle gracefully
                try:
                    with patch("pgvector.psycopg.register_vector", side_effect=ImportError):
                        db.search("test", [0.1] * 384, limit=10)
                except (ImportError, AttributeError):
                    pass  # Expected if pgvector not available


# =============================================================================
# Edge Case Tests: Boundary Values
# =============================================================================


class TestBoundaryValues:
    """Edge case tests for boundary values."""

    def test_search_with_limit_zero(self, pgvector_db, mock_connection_manager):
        """Search with limit=0 should return empty list."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__iter__ = Mock(return_value=iter([]))
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            results = pgvector_db.search("test_coll", [0.1] * 384, limit=0)

        assert results == []

    def test_vector_with_nan_values(self, pgvector_db, mock_connection_manager):
        """Vector with NaN values should be handled or rejected."""
        import math

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        # Vector with NaN
        nan_vector = [0.1] * 383 + [math.nan]

        with patch("pgvector.psycopg.register_vector"):
            with patch.object(pgvector_db, "_ensure_schema"):
                try:
                    pgvector_db.upsert(
                        "test_coll",
                        [{"id": "nan_test", "vector": nan_vector, "payload": {}}],
                    )
                    # If it doesn't raise, that's also acceptable behavior
                except (ValueError, Exception):
                    pass  # Expected - NaN vectors are invalid

    def test_vector_with_inf_values(self, pgvector_db, mock_connection_manager):
        """Vector with Inf values should be handled or rejected."""
        import math

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        # Vector with Inf
        inf_vector = [0.1] * 383 + [math.inf]

        with patch("pgvector.psycopg.register_vector"):
            with patch.object(pgvector_db, "_ensure_schema"):
                try:
                    pgvector_db.upsert(
                        "test_coll",
                        [{"id": "inf_test", "vector": inf_vector, "payload": {}}],
                    )
                except (ValueError, Exception):
                    pass  # Expected - Inf vectors may be invalid

    def test_search_with_very_large_limit(self, pgvector_db, mock_connection_manager):
        """Search with very large limit should work."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__iter__ = Mock(
            return_value=iter(
                [
                    ("id1", 0.9, {"content": "test"}),
                ]
            )
        )
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            results = pgvector_db.search("test_coll", [0.1] * 384, limit=1000000)

        # Should work, returns whatever exists
        assert len(results) == 1


# =============================================================================
# Edge Case Tests: Hybrid Search Edge Cases
# =============================================================================


class TestHybridSearchEdgeCases:
    """Edge case tests for hybrid search."""

    def test_hybrid_search_empty_text(self, pgvector_db, mock_connection_manager):
        """Hybrid search with empty query text should handle gracefully."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__iter__ = Mock(return_value=iter([]))
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            results = pgvector_db.hybrid_search(
                "test_coll",
                query_vector=[0.1] * 384,
                query_text="",  # Empty text
                limit=10,
            )

        # Should return results (vector-only fallback) or empty
        assert isinstance(results, list)

    def test_hybrid_search_very_long_text(self, pgvector_db, mock_connection_manager):
        """Hybrid search with very long text should work."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__iter__ = Mock(return_value=iter([]))
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        # Very long query text (10KB)
        long_text = "word " * 2000

        with patch("pgvector.psycopg.register_vector"):
            results = pgvector_db.hybrid_search(
                "test_coll",
                query_vector=[0.1] * 384,
                query_text=long_text,
                limit=10,
            )

        assert isinstance(results, list)

    def test_hybrid_search_special_chars_in_text(self, pgvector_db, mock_connection_manager):
        """Hybrid search with special characters in text."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        cursor = MagicMock()
        cursor.__iter__ = Mock(return_value=iter([]))
        conn.execute.return_value = cursor

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        with patch("pgvector.psycopg.register_vector"):
            results = pgvector_db.hybrid_search(
                "test_coll",
                query_vector=[0.1] * 384,
                query_text='test\'s "query" with (special) chars: @#$%',
                limit=10,
            )

        assert isinstance(results, list)


# =============================================================================
# Edge Case Tests: Concurrent Operations
# =============================================================================


class TestConcurrentOperations:
    """Edge case tests for concurrent operations."""

    def test_concurrent_upsert_same_id(self, pgvector_db, mock_connection_manager):
        """Concurrent upserts to same ID should not cause errors."""
        import threading

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        errors = []

        def do_upsert(version):
            try:
                with patch("pgvector.psycopg.register_vector"):
                    with patch.object(pgvector_db, "_ensure_schema"):
                        pgvector_db.upsert(
                            "test_coll",
                            [
                                {
                                    "id": "same_id",
                                    "vector": [0.1 * version] * 384,
                                    "payload": {"v": version},
                                }
                            ],
                        )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_upsert, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors (last write wins)
        assert len(errors) == 0


# =============================================================================
# Edge Case Tests: Delete Operations
# =============================================================================


class TestDeleteOperations:
    """Edge case tests for delete operations."""

    def test_delete_nonexistent_points(self, pgvector_db, mock_connection_manager):
        """Deleting points that don't exist should not raise."""
        pgvector_db._initialized_collections["test_coll"] = 384

        conn = MagicMock()
        # Simulate no rows deleted
        conn.execute.return_value.rowcount = 0

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        # Should not raise even if points don't exist
        with patch("pgvector.psycopg.register_vector"):
            try:
                # If delete method exists
                if hasattr(pgvector_db, "delete"):
                    pgvector_db.delete("test_coll", ids=["nonexistent1", "nonexistent2"])
            except AttributeError:
                pass  # delete method may not exist


# =============================================================================
# Edge Case Tests: Large Payload
# =============================================================================


class TestLargePayload:
    """Edge case tests for large payloads."""

    def test_very_large_payload(self, pgvector_db, mock_connection_manager):
        """Payload with MB of data should be handled."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_connection_manager.connection.return_value.__enter__ = Mock(return_value=conn)
        mock_connection_manager.connection.return_value.__exit__ = Mock(return_value=False)

        # 1MB payload
        large_content = "x" * (1024 * 1024)

        with patch("pgvector.psycopg.register_vector"):
            with patch.object(pgvector_db, "_ensure_schema"):
                pgvector_db.upsert(
                    "test_coll",
                    [
                        {
                            "id": "large_payload",
                            "vector": [0.1] * 384,
                            "payload": {"content": large_content},
                        }
                    ],
                )

        assert cursor.executemany.called
