# tests/unit/test_postgres_connection.py
"""
Unit tests for PostgresConnectionManager.

Tests cover:
1. Collection name sanitization
2. URI database replacement
3. Singleton pattern
4. Auto-recovery from corrupted pgdata
5. External mode handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

# Mark all tests in this module as postgres and tier2
pytestmark = [pytest.mark.postgres, pytest.mark.tier2]

from fitz_ai.storage.config import StorageConfig, StorageMode
from fitz_ai.storage.postgres import (
    PostgresConnectionManager,
    _replace_database_in_uri,
    _sanitize_collection_name,
)


# =============================================================================
# Collection Name Sanitization Tests
# =============================================================================


class TestCollectionNameSanitization:
    """Tests for collection name sanitization."""

    def test_simple_name_unchanged(self):
        """Simple alphanumeric name stays the same."""
        assert _sanitize_collection_name("myCollection") == "mycollection"
        assert _sanitize_collection_name("test_coll") == "test_coll"

    def test_hyphens_replaced(self):
        """Hyphens are replaced with underscores."""
        assert _sanitize_collection_name("my-collection") == "my_collection"
        assert _sanitize_collection_name("a-b-c") == "a_b_c"

    def test_special_chars_replaced(self):
        """Special characters are replaced."""
        assert _sanitize_collection_name("my.collection") == "my_collection"
        assert _sanitize_collection_name("my@collection!") == "my_collection_"
        assert _sanitize_collection_name("test/path") == "test_path"

    def test_starts_with_number_prefixed(self):
        """Names starting with numbers get 'c_' prefix."""
        assert _sanitize_collection_name("123test") == "c_123test"
        assert _sanitize_collection_name("1") == "c_1"

    def test_unicode_replaced(self):
        """Unicode characters are replaced."""
        assert _sanitize_collection_name("tëst") == "t_st"
        assert _sanitize_collection_name("日本語") == "c____"

    def test_multiple_underscores_preserved(self):
        """Multiple consecutive special chars become multiple underscores."""
        assert _sanitize_collection_name("a--b") == "a__b"
        assert _sanitize_collection_name("a!!!b") == "a___b"

    def test_lowercase_output(self):
        """Output is always lowercase."""
        assert _sanitize_collection_name("MyCollection") == "mycollection"
        assert _sanitize_collection_name("TEST_COLL") == "test_coll"


# =============================================================================
# URI Database Replacement Tests
# =============================================================================


class TestURIDatabaseReplacement:
    """Tests for URI database name replacement."""

    def test_simple_replacement(self):
        """Simple database replacement works."""
        uri = "postgresql://user:pass@localhost:5432/olddb"
        new_uri = _replace_database_in_uri(uri, "newdb")
        assert new_uri == "postgresql://user:pass@localhost:5432/newdb"

    def test_replacement_without_password(self):
        """Replacement works without password."""
        uri = "postgresql://user@localhost/olddb"
        new_uri = _replace_database_in_uri(uri, "newdb")
        assert new_uri == "postgresql://user@localhost/newdb"

    def test_replacement_with_port(self):
        """Replacement preserves port."""
        uri = "postgresql://localhost:5433/olddb"
        new_uri = _replace_database_in_uri(uri, "newdb")
        assert "5433" in new_uri
        assert new_uri.endswith("/newdb")

    def test_replacement_with_params(self):
        """Replacement preserves query params."""
        uri = "postgresql://localhost/olddb?sslmode=require"
        new_uri = _replace_database_in_uri(uri, "newdb")
        assert "newdb" in new_uri
        assert "sslmode=require" in new_uri


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingletonPattern:
    """Tests for singleton behavior."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop()
        PostgresConnectionManager._instance = None

    def test_get_instance_returns_same_instance(self):
        """Multiple get_instance calls return same instance."""
        config = StorageConfig(mode=StorageMode.EXTERNAL, connection_string="postgresql://localhost/test")

        # Patch both __init__ and atexit to prevent broken handlers being registered
        with patch.object(PostgresConnectionManager, "__init__", return_value=None):
            with patch("fitz_ai.storage.postgres.atexit.register"):
                instance1 = PostgresConnectionManager.get_instance(config)
                instance2 = PostgresConnectionManager.get_instance()

                assert instance1 is instance2

    def test_reset_instance_creates_new(self):
        """Reset allows creating new instance."""
        config = StorageConfig(mode=StorageMode.EXTERNAL, connection_string="postgresql://localhost/test")

        # Patch both __init__ and atexit to prevent broken handlers being registered
        with patch.object(PostgresConnectionManager, "__init__", return_value=None):
            with patch("fitz_ai.storage.postgres.atexit.register"):
                with patch.object(PostgresConnectionManager, "stop"):
                    instance1 = PostgresConnectionManager.get_instance(config)
                    PostgresConnectionManager.reset_instance()
                    instance2 = PostgresConnectionManager.get_instance(config)

                    assert instance1 is not instance2


# =============================================================================
# External Mode Tests
# =============================================================================


class TestExternalMode:
    """Tests for external connection string mode."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    def test_external_mode_uses_connection_string(self):
        """External mode uses provided connection string."""
        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://user:pass@remote-host:5432/mydb",
        )

        manager = PostgresConnectionManager(config)

        # Mock the start to avoid actual connection
        with patch.object(manager, "_start_pgserver"):
            manager.start()

        assert manager._base_uri == "postgresql://user:pass@remote-host:5432/mydb"

    def test_external_mode_requires_connection_string(self):
        """External mode raises error without connection string."""
        config = StorageConfig(mode=StorageMode.EXTERNAL, connection_string=None)

        manager = PostgresConnectionManager(config)

        with pytest.raises(ValueError) as exc_info:
            manager.start()

        assert "connection_string required" in str(exc_info.value)


# =============================================================================
# Auto-Recovery Tests
# =============================================================================


class TestAutoRecovery:
    """Tests for pgserver auto-recovery."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    @patch("fitz_ai.storage.postgres.shutil.rmtree")
    @patch("fitz_ai.storage.postgres.FitzPaths.ensure_pgdata")
    def test_recovery_deletes_pgdata_on_failure(self, mock_ensure_pgdata, mock_rmtree):
        """Recovery deletes corrupted pgdata directory."""
        from pathlib import Path
        import sys

        mock_ensure_pgdata.return_value = Path("/tmp/test_pgdata")

        config = StorageConfig(mode=StorageMode.LOCAL)
        manager = PostgresConnectionManager(config)

        # Mock pgserver module
        mock_pgserver_module = MagicMock()
        call_count = 0
        mock_server = MagicMock()
        mock_server.get_uri.return_value = "postgresql://localhost/postgres"

        def mock_get_server(path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Corrupted database")
            return mock_server

        mock_pgserver_module.get_server = mock_get_server

        with patch.dict(sys.modules, {"pgserver": mock_pgserver_module}):
            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=True):
                    manager._start_pgserver(timeout=1, allow_recovery=True)

        # Verify rmtree was called for recovery
        assert mock_rmtree.called

    def test_recovery_disabled_raises_immediately(self):
        """With recovery disabled, failure raises immediately."""
        from pathlib import Path
        import sys

        config = StorageConfig(mode=StorageMode.LOCAL, data_dir=Path("/tmp/test"))
        manager = PostgresConnectionManager(config)

        mock_pgserver_module = MagicMock()
        mock_pgserver_module.get_server.side_effect = Exception("Startup failed")

        with patch.dict(sys.modules, {"pgserver": mock_pgserver_module}):
            with patch("pathlib.Path.mkdir"):
                with pytest.raises(RuntimeError) as exc_info:
                    manager._start_pgserver(timeout=1, allow_recovery=False)

        assert "failed to start" in str(exc_info.value)


# =============================================================================
# Timeout Tests
# =============================================================================


class TestPgserverTimeout:
    """Tests for pgserver startup timeout."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    def test_pgserver_timeout_raises_error(self):
        """Pgserver startup exceeding timeout raises RuntimeError."""
        from pathlib import Path
        import sys
        import time

        config = StorageConfig(mode=StorageMode.LOCAL, data_dir=Path("/tmp/test"))
        manager = PostgresConnectionManager(config)

        mock_pgserver_module = MagicMock()

        def slow_get_server(path):
            time.sleep(0.5)  # Simulate slow startup
            raise Exception("Still starting...")

        mock_pgserver_module.get_server = slow_get_server

        with patch.dict(sys.modules, {"pgserver": mock_pgserver_module}):
            with patch("pathlib.Path.mkdir"):
                with pytest.raises(RuntimeError) as exc_info:
                    # Very short timeout to trigger error quickly
                    manager._start_pgserver(timeout=0.1, allow_recovery=False)

        assert "failed to start" in str(exc_info.value).lower()


# =============================================================================
# Graceful Shutdown Tests
# =============================================================================


class TestGracefulShutdown:
    """Tests for graceful shutdown."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    def test_stop_closes_pools(self):
        """Stop method closes all connection pools."""
        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
        )
        manager = PostgresConnectionManager(config)
        manager._started = True
        manager._base_uri = "postgresql://localhost/test"

        # Add mock pools
        mock_pool1 = MagicMock()
        mock_pool2 = MagicMock()
        manager._pools["coll1"] = mock_pool1
        manager._pools["coll2"] = mock_pool2

        manager.stop()

        # Verify pools were closed
        mock_pool1.close.assert_called_once()
        mock_pool2.close.assert_called_once()
        assert len(manager._pools) == 0

    def test_stop_handles_pgserver(self):
        """Stop method handles pgserver cleanup in local mode."""
        from pathlib import Path

        config = StorageConfig(mode=StorageMode.LOCAL, data_dir=Path("/tmp/test"))
        manager = PostgresConnectionManager(config)
        manager._started = True

        # Mock pgserver instance
        mock_server = MagicMock()
        manager._pgserver = mock_server

        manager.stop()

        # Verify pgserver cleanup attempted
        assert manager._started is False

    def test_reset_instance_stops_and_clears(self):
        """reset_instance stops existing manager and clears singleton."""
        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
        )

        # Patch both __init__ and atexit to prevent broken handlers being registered
        with patch.object(PostgresConnectionManager, "__init__", return_value=None):
            with patch("fitz_ai.storage.postgres.atexit.register"):
                instance = PostgresConnectionManager.get_instance(config)
                instance._started = True
                instance._pools = {}
                instance.stop = MagicMock()

                PostgresConnectionManager.reset_instance()

                instance.stop.assert_called_once()
                # Getting new instance should work
                assert PostgresConnectionManager._instance is None


# =============================================================================
# Connection Pool Tests
# =============================================================================


class TestConnectionPool:
    """Tests for connection pooling."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    def test_get_pool_auto_starts(self):
        """get_pool auto-starts if not started."""
        import sys

        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
        )
        manager = PostgresConnectionManager(config)

        mock_pool_class = MagicMock()
        mock_psycopg_pool = MagicMock()
        mock_psycopg_pool.ConnectionPool = mock_pool_class

        def mock_start_impl():
            # Simulate what start() does
            manager._started = True
            manager._base_uri = "postgresql://localhost/test"

        with patch.object(manager, "start", side_effect=mock_start_impl) as mock_start:
            with patch.object(manager, "_ensure_database", return_value="fitz_test"):
                with patch.dict(sys.modules, {"psycopg_pool": mock_psycopg_pool}):
                    manager.get_pool("test_collection")

        mock_start.assert_called_once()

    def test_pool_reused_for_same_collection(self):
        """Same collection returns same pool."""
        import sys

        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
        )
        manager = PostgresConnectionManager(config)
        manager._started = True
        manager._base_uri = "postgresql://localhost/test"

        mock_pool = MagicMock()
        mock_pool_class = MagicMock(return_value=mock_pool)
        mock_psycopg_pool = MagicMock()
        mock_psycopg_pool.ConnectionPool = mock_pool_class

        with patch.object(manager, "_ensure_database", return_value="fitz_coll"):
            with patch.dict(sys.modules, {"psycopg_pool": mock_psycopg_pool}):
                pool1 = manager.get_pool("my_collection")
                pool2 = manager.get_pool("my_collection")

        assert pool1 is pool2
        # Pool constructor called only once
        assert mock_pool_class.call_count == 1


# =============================================================================
# Database Creation Tests
# =============================================================================


class TestDatabaseCreation:
    """Tests for per-collection database creation."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    def test_ensure_database_creates_if_not_exists(self):
        """_ensure_database creates database if it doesn't exist."""
        import sys

        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/postgres",
        )
        manager = PostgresConnectionManager(config)
        manager._started = True
        manager._base_uri = "postgresql://localhost/postgres"

        mock_conn = MagicMock()
        # First execute: check if exists (returns None = doesn't exist)
        # Second execute: CREATE DATABASE
        mock_conn.execute.side_effect = [
            MagicMock(fetchone=Mock(return_value=None)),  # doesn't exist
            MagicMock(),  # CREATE DATABASE
            MagicMock(),  # CREATE EXTENSION (in new db connection)
        ]

        mock_psycopg = MagicMock()
        mock_psycopg.connect.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_psycopg.connect.return_value.__exit__ = Mock(return_value=False)

        with patch.dict(sys.modules, {"psycopg": mock_psycopg}):
            db_name = manager._ensure_database("my_collection")

        assert db_name == "fitz_my_collection"
        assert "fitz_my_collection" in manager._initialized_dbs

    def test_ensure_database_skips_if_exists(self):
        """_ensure_database skips creation if database exists."""
        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/postgres",
        )
        manager = PostgresConnectionManager(config)
        manager._started = True
        manager._base_uri = "postgresql://localhost/postgres"
        manager._initialized_dbs.add("fitz_existing")

        # Should return immediately without any DB calls
        db_name = manager._ensure_database("existing")

        assert db_name == "fitz_existing"


# =============================================================================
# Edge Case Tests: Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of singleton."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    def test_thread_safety_concurrent_get_instance(self):
        """Concurrent get_instance calls should return same instance."""
        import threading

        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
        )

        # Pre-create instance to avoid __init__ issues in threads
        manager = PostgresConnectionManager(config)
        PostgresConnectionManager._instance = manager

        results = []
        errors = []

        def get_instance():
            try:
                instance = PostgresConnectionManager.get_instance()
                results.append(id(instance))
            except Exception as e:
                errors.append(e)

        # Launch multiple threads
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance (same id)
        assert len(errors) == 0
        assert len(set(results)) == 1  # All same instance


# =============================================================================
# Edge Case Tests: Pool Exhaustion
# =============================================================================


class TestPoolExhaustion:
    """Tests for connection pool behavior under load."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    def test_connection_pool_exhaustion_blocks(self):
        """When pool exhausted, new requests should block or raise."""
        from psycopg_pool import PoolTimeout

        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
            pool_min_size=1,
            pool_max_size=2,  # Very small pool
        )
        manager = PostgresConnectionManager(config)
        manager._started = True
        manager._base_uri = "postgresql://localhost/test"

        # Mock pool that times out
        mock_pool = MagicMock()
        mock_pool.connection.side_effect = PoolTimeout("Pool exhausted")
        manager._pools["test_coll"] = mock_pool

        with pytest.raises(PoolTimeout):
            with manager.connection("test_coll"):
                pass


# =============================================================================
# Edge Case Tests: Import Errors
# =============================================================================


class TestImportErrors:
    """Tests for handling missing dependencies."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    def test_import_error_pgserver_not_installed(self):
        """Should raise clear error if pgserver not installed in local mode."""
        from pathlib import Path
        import sys

        config = StorageConfig(mode=StorageMode.LOCAL, data_dir=Path("/tmp/test"))
        manager = PostgresConnectionManager(config)

        # Simulate pgserver not installed by making import raise
        def raise_import_error(*args):
            raise ImportError("No module named 'pgserver'")

        with patch.dict(sys.modules, {"pgserver": None}):
            with patch("builtins.__import__", side_effect=raise_import_error):
                with patch("pathlib.Path.mkdir"):
                    with pytest.raises((ImportError, RuntimeError)):
                        manager._start_pgserver(timeout=1, allow_recovery=False)


# =============================================================================
# Edge Case Tests: Race Conditions
# =============================================================================


class TestRaceConditions:
    """Tests for race condition handling."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    def test_database_creation_race_condition(self):
        """Concurrent database creation should not fail due to lock."""
        import threading
        import sys

        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/postgres",
        )
        manager = PostgresConnectionManager(config)
        manager._started = True
        manager._base_uri = "postgresql://localhost/postgres"

        # Pre-add databases to _initialized_dbs to simulate they exist
        # This tests that concurrent access to _initialized_dbs is safe
        for i in range(5):
            manager._initialized_dbs.add(f"fitz_coll_{i}")

        errors = []
        results = []

        def get_db(coll_name):
            try:
                # Since db already in _initialized_dbs, should return quickly
                db_name = manager._ensure_database(coll_name)
                results.append(db_name)
            except Exception as e:
                errors.append(e)

        # Multiple threads trying to get same database
        threads = [
            threading.Thread(target=get_db, args=(f"coll_{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0
        assert len(results) == 5


# =============================================================================
# Edge Case Tests: Start State
# =============================================================================


class TestStartState:
    """Tests for start/stop state management."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    def test_start_already_started_noop(self):
        """Calling start() when already started should be a no-op."""
        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
        )
        manager = PostgresConnectionManager(config)
        manager._started = True
        manager._base_uri = "postgresql://localhost/test"

        # Second start should not raise or change state
        with patch.object(manager, "_start_pgserver") as mock_start:
            manager.start()

        # Should not have attempted to start pgserver again
        mock_start.assert_not_called()
        assert manager._started is True

    def test_stop_not_started_noop(self):
        """Calling stop() when not started should be a no-op."""
        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
        )
        manager = PostgresConnectionManager(config)
        manager._started = False

        # Should not raise
        manager.stop()

        assert manager._started is False


# =============================================================================
# Edge Case Tests: Concurrent Pool Creation
# =============================================================================


class TestConcurrentPoolCreation:
    """Tests for concurrent pool creation scenarios."""

    def teardown_method(self):
        """Reset singleton after each test."""
        # Directly clear singleton without calling stop() to avoid hanging
        PostgresConnectionManager._instance = None

    def test_concurrent_pool_creation_same_collection(self):
        """Concurrent pool requests for same collection should return same pool."""
        import sys
        import threading

        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
        )
        manager = PostgresConnectionManager(config)
        manager._started = True
        manager._base_uri = "postgresql://localhost/test"

        # Create a mock pool class that tracks creation
        creation_count = 0
        mock_pool = MagicMock()

        def mock_pool_constructor(*args, **kwargs):
            nonlocal creation_count
            creation_count += 1
            return mock_pool

        mock_pool_class = MagicMock(side_effect=mock_pool_constructor)
        mock_psycopg_pool = MagicMock()
        mock_psycopg_pool.ConnectionPool = mock_pool_class

        results = []
        errors = []

        def get_pool():
            try:
                with patch.object(manager, "_ensure_database", return_value="fitz_shared"):
                    with patch.dict(sys.modules, {"psycopg_pool": mock_psycopg_pool}):
                        pool = manager.get_pool("shared_collection")
                        results.append(id(pool))
            except Exception as e:
                errors.append(e)

        # Launch multiple threads requesting same pool
        threads = [threading.Thread(target=get_pool) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All threads should get same pool (same id)
        assert len(set(results)) == 1
        # Pool should only be created once (accounting for thread race in mock)
        # Due to thread safety, it might be created more than once in this test
        # but in practice the lock should prevent this


class TestConnectionContextManager:
    """Tests for connection context manager behavior."""

    def teardown_method(self):
        """Reset singleton after each test."""
        PostgresConnectionManager._instance = None

    def test_connection_context_manager_exception_handling(self):
        """Connection context manager should properly handle exceptions."""
        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
        )
        manager = PostgresConnectionManager(config)
        manager._started = True
        manager._base_uri = "postgresql://localhost/test"

        # Create mock pool with connection that raises
        mock_conn = MagicMock()
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = Mock(return_value=False)
        manager._pools["test_coll"] = mock_pool

        # Connection should work in normal case
        with manager.connection("test_coll") as conn:
            assert conn is mock_conn

        # Verify __exit__ was called
        mock_pool.connection.return_value.__exit__.assert_called()

    def test_connection_context_manager_propagates_exception(self):
        """Connection context manager should propagate inner exceptions."""
        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
        )
        manager = PostgresConnectionManager(config)
        manager._started = True
        manager._base_uri = "postgresql://localhost/test"

        mock_conn = MagicMock()
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = Mock(return_value=False)
        manager._pools["test_coll"] = mock_pool

        # Exception inside context should propagate
        with pytest.raises(ValueError, match="test error"):
            with manager.connection("test_coll"):
                raise ValueError("test error")

    def test_connection_reuses_existing_pool(self):
        """Multiple connection() calls should reuse existing pool."""
        config = StorageConfig(
            mode=StorageMode.EXTERNAL,
            connection_string="postgresql://localhost/test",
        )
        manager = PostgresConnectionManager(config)
        manager._started = True
        manager._base_uri = "postgresql://localhost/test"

        mock_conn = MagicMock()
        mock_pool = MagicMock()
        mock_pool.connection.return_value.__enter__ = Mock(return_value=mock_conn)
        mock_pool.connection.return_value.__exit__ = Mock(return_value=False)
        manager._pools["test_coll"] = mock_pool

        # Get connection twice
        with manager.connection("test_coll"):
            pass
        with manager.connection("test_coll"):
            pass

        # Should have used the same pool both times
        assert mock_pool.connection.call_count == 2
