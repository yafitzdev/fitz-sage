# tests/unit/test_postgres_recovery.py
"""
Tests for PostgreSQL connection manager recovery logic.

Tests both unit-level helper functions and integration-level
recovery scenarios to ensure pgserver handles failures gracefully.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fitz_ai.storage.postgres import (
    MAX_RESTART_ATTEMPTS,
    RETRY_BACKOFF_BASE,
    RETRY_BACKOFF_MAX,
    PostgresConnectionManager,
    _calculate_backoff,
    _cleanup_stale_lock_files,
    _force_remove_pgdata,
    _kill_zombie_postgres_processes,
    _sanitize_collection_name,
)
from fitz_ai.storage.config import StorageConfig, StorageMode


# =============================================================================
# Unit Tests - Helper Functions
# =============================================================================


class TestCalculateBackoff:
    """Tests for exponential backoff calculation."""

    def test_first_attempt_backoff(self):
        """First attempt uses base backoff."""
        backoff = _calculate_backoff(0)
        assert backoff == RETRY_BACKOFF_BASE

    def test_exponential_growth(self):
        """Backoff grows exponentially."""
        backoff_0 = _calculate_backoff(0)
        backoff_1 = _calculate_backoff(1)
        backoff_2 = _calculate_backoff(2)

        assert backoff_1 == backoff_0 * 2
        assert backoff_2 == backoff_0 * 4

    def test_max_backoff_cap(self):
        """Backoff is capped at maximum."""
        # Large attempt number should hit cap
        backoff = _calculate_backoff(100)
        assert backoff == RETRY_BACKOFF_MAX

    def test_backoff_never_exceeds_max(self):
        """No backoff value exceeds maximum."""
        for attempt in range(20):
            backoff = _calculate_backoff(attempt)
            assert backoff <= RETRY_BACKOFF_MAX


class TestSanitizeCollectionName:
    """Tests for collection name sanitization."""

    def test_valid_name_unchanged(self):
        """Valid names pass through."""
        assert _sanitize_collection_name("my_collection") == "my_collection"
        assert _sanitize_collection_name("Test123") == "test123"

    def test_invalid_chars_replaced(self):
        """Invalid characters become underscores."""
        assert _sanitize_collection_name("my-collection") == "my_collection"
        assert _sanitize_collection_name("my.collection") == "my_collection"
        assert _sanitize_collection_name("my collection") == "my_collection"

    def test_numeric_start_prefixed(self):
        """Names starting with numbers get prefix."""
        result = _sanitize_collection_name("123test")
        assert result.startswith("c_")
        assert "123test" in result

    def test_result_is_lowercase(self):
        """Result is always lowercase."""
        assert _sanitize_collection_name("MyCollection") == "mycollection"


class TestCleanupStaleLockFiles:
    """Tests for stale lock file cleanup."""

    def test_removes_postmaster_pid(self):
        """Removes postmaster.pid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            lock_file = data_dir / "postmaster.pid"
            lock_file.write_text("12345")

            _cleanup_stale_lock_files(data_dir)

            assert not lock_file.exists()

    def test_removes_socket_lock(self):
        """Removes socket lock files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            lock_file = data_dir / ".s.PGSQL.5432.lock"
            lock_file.write_text("lock")

            _cleanup_stale_lock_files(data_dir)

            assert not lock_file.exists()

    def test_removes_socket_files(self):
        """Removes socket files matching pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            socket_file = data_dir / ".s.PGSQL.5432"
            socket_file.write_text("socket")

            _cleanup_stale_lock_files(data_dir)

            assert not socket_file.exists()

    def test_handles_missing_files(self):
        """Doesn't fail if files don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            # No lock files - should not raise
            _cleanup_stale_lock_files(data_dir)

    def test_handles_missing_directory(self):
        """Doesn't fail if directory doesn't exist."""
        nonexistent = Path("/nonexistent/path/that/does/not/exist")
        # Should not raise (glob on nonexistent dir returns empty)
        _cleanup_stale_lock_files(nonexistent)


class TestForceRemovePgdata:
    """Tests for forced pgdata removal."""

    def test_removes_empty_directory(self):
        """Removes empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "pgdata"
            data_dir.mkdir()

            result = _force_remove_pgdata(data_dir)

            assert result is True
            assert not data_dir.exists()

    def test_removes_directory_with_files(self):
        """Removes directory with files inside."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "pgdata"
            data_dir.mkdir()
            (data_dir / "file1.txt").write_text("content")
            (data_dir / "subdir").mkdir()
            (data_dir / "subdir" / "file2.txt").write_text("more content")

            result = _force_remove_pgdata(data_dir)

            assert result is True
            assert not data_dir.exists()

    def test_returns_true_for_nonexistent(self):
        """Returns True if directory doesn't exist."""
        nonexistent = Path("/nonexistent/pgdata")
        result = _force_remove_pgdata(nonexistent)
        assert result is True

    def test_retries_on_permission_error(self):
        """Retries when permission error occurs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "pgdata"
            data_dir.mkdir()

            call_count = 0
            original_rmtree = shutil.rmtree

            def mock_rmtree(path, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise PermissionError("File in use")
                return original_rmtree(path, *args, **kwargs)

            with patch("shutil.rmtree", side_effect=mock_rmtree):
                with patch(
                    "fitz_ai.storage.postgres._kill_zombie_postgres_processes"
                ):
                    result = _force_remove_pgdata(data_dir, max_retries=5)

            # Should have retried and eventually succeeded
            assert call_count >= 3


class TestKillZombiePostgresProcesses:
    """Tests for zombie process killing."""

    def test_does_not_raise_when_no_processes(self):
        """Doesn't fail when no postgres processes exist."""
        # Should complete without raising
        _kill_zombie_postgres_processes()

    @patch("subprocess.run")
    def test_calls_taskkill_on_windows(self, mock_run):
        """Calls taskkill on Windows."""
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.platform", "win32"):
            _kill_zombie_postgres_processes()

        # Should have called taskkill for postgres processes
        assert mock_run.called

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Unix-specific test (os.getuid not available on Windows)"
    )
    @patch("subprocess.run")
    def test_calls_pkill_on_unix(self, mock_run):
        """Calls pkill on Unix."""
        mock_run.return_value = MagicMock(returncode=0)

        with patch("sys.platform", "linux"):
            _kill_zombie_postgres_processes()

        assert mock_run.called


# =============================================================================
# Integration Tests - Recovery Scenarios
# =============================================================================


@pytest.mark.postgres
class TestPgserverRecoveryIntegration:
    """Integration tests for pgserver recovery.

    These tests create actual stuck states and verify recovery.
    Marked as postgres tests - run separately from parallel tests.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Reset singleton before and after each test."""
        PostgresConnectionManager.reset_instance()
        yield
        PostgresConnectionManager.reset_instance()

    @pytest.fixture
    def temp_pgdata(self):
        """Create a temporary pgdata directory."""
        import time
        tmpdir = tempfile.mkdtemp()
        pgdata = Path(tmpdir) / "pgdata"
        pgdata.mkdir()
        yield pgdata
        # Cleanup with retry for Windows file handle issues
        for _ in range(5):
            try:
                shutil.rmtree(tmpdir)
                break
            except PermissionError:
                _kill_zombie_postgres_processes()
                time.sleep(1.0)
        # Ignore if cleanup fails - temp dir will be cleaned by OS eventually

    def test_recovery_from_stale_lock_file(self, temp_pgdata):
        """Pgserver recovers when stale lock file exists."""
        # Create stale lock file with fake PID
        lock_file = temp_pgdata / "postmaster.pid"
        lock_file.write_text("99999\n/fake/path\n12345\n")  # Fake content

        # Create manager with temp pgdata
        config = StorageConfig(
            mode=StorageMode.LOCAL,
            data_dir=str(temp_pgdata),
        )
        manager = PostgresConnectionManager(config)

        try:
            # Should start successfully (recovery cleans up stale lock file)
            manager.start()

            assert manager._started is True
            assert manager._pgserver is not None

            # Verify we can actually connect
            is_healthy, _ = manager.is_healthy()
            assert is_healthy is True
        finally:
            manager.stop()
            # Give Windows time to release file handles
            import time
            time.sleep(1.0)

    def test_recovery_cleans_socket_files(self, temp_pgdata):
        """Pgserver cleans up stale socket files."""
        # Create stale socket files
        (temp_pgdata / ".s.PGSQL.5432").write_text("socket")
        (temp_pgdata / ".s.PGSQL.5432.lock").write_text("lock")

        config = StorageConfig(
            mode=StorageMode.LOCAL,
            data_dir=str(temp_pgdata),
        )
        manager = PostgresConnectionManager(config)

        try:
            manager.start()
            assert manager._started is True
        finally:
            manager.stop()
            time.sleep(1.0)

    def test_connection_creates_database(self, temp_pgdata):
        """Connection creates database if it doesn't exist."""
        config = StorageConfig(
            mode=StorageMode.LOCAL,
            data_dir=str(temp_pgdata),
        )
        manager = PostgresConnectionManager(config)

        try:
            manager.start()

            # Get connection for a new collection
            with manager.connection("test_collection") as conn:
                result = conn.execute("SELECT 1").fetchone()
                assert result[0] == 1
        finally:
            manager.stop()
            time.sleep(1.0)

    def test_connection_recovery_after_restart(self, temp_pgdata):
        """Connection recovers after pgserver restart."""
        config = StorageConfig(
            mode=StorageMode.LOCAL,
            data_dir=str(temp_pgdata),
        )
        manager = PostgresConnectionManager(config)

        try:
            manager.start()

            # First connection works
            with manager.connection("test_recovery") as conn:
                conn.execute("SELECT 1")

            # Simulate restart
            manager._restart_pgserver()

            # Connection should still work after restart
            with manager.connection("test_recovery") as conn:
                result = conn.execute("SELECT 1").fetchone()
                assert result[0] == 1
        finally:
            manager.stop()
            time.sleep(1.0)

    def test_health_check_after_start(self, temp_pgdata):
        """Health check returns healthy after successful start."""
        config = StorageConfig(
            mode=StorageMode.LOCAL,
            data_dir=str(temp_pgdata),
        )
        manager = PostgresConnectionManager(config)

        try:
            manager.start()

            is_healthy, message = manager.is_healthy()

            assert is_healthy is True
            assert "Connected" in message
        finally:
            manager.stop()
            time.sleep(1.0)

    def test_stats_tracking(self, temp_pgdata):
        """Stats are tracked correctly."""
        config = StorageConfig(
            mode=StorageMode.LOCAL,
            data_dir=str(temp_pgdata),
        )
        manager = PostgresConnectionManager(config)

        try:
            manager.start()

            # Create some connections
            with manager.connection("collection_a") as conn:
                conn.execute("SELECT 1")
            with manager.connection("collection_b") as conn:
                conn.execute("SELECT 1")

            stats = manager.get_stats()

            assert stats["started"] is True
            assert stats["mode"] == "local"
            assert stats["pools"] == 2
            assert stats["databases"] >= 2
        finally:
            manager.stop()
            time.sleep(1.0)


@pytest.mark.postgres
class TestPgserverMaxRetries:
    """Test that max retry limit is enforced."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Reset singleton before and after each test."""
        PostgresConnectionManager.reset_instance()
        yield
        PostgresConnectionManager.reset_instance()

    def test_restart_fails_after_max_attempts(self):
        """Restart raises error after max attempts exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(
                mode=StorageMode.LOCAL,
                data_dir=str(Path(tmpdir) / "pgdata"),
            )
            manager = PostgresConnectionManager(config)

            # Set restart attempts to max
            manager._restart_attempts = MAX_RESTART_ATTEMPTS

            # Next restart should fail
            with pytest.raises(RuntimeError, match="restart failed"):
                manager._restart_pgserver()

    def test_restart_counter_resets_on_success(self):
        """Restart counter resets after successful restart."""
        tmpdir = tempfile.mkdtemp()
        try:
            pgdata = Path(tmpdir) / "pgdata"
            pgdata.mkdir()

            config = StorageConfig(
                mode=StorageMode.LOCAL,
                data_dir=str(pgdata),
            )
            manager = PostgresConnectionManager(config)
            manager.start()

            # Simulate some restart attempts
            manager._restart_attempts = 2

            # Successful restart should reset counter
            manager._restart_pgserver()

            assert manager._restart_attempts == 0

            manager.stop()
            time.sleep(1.0)
        finally:
            # Cleanup with retry
            for _ in range(5):
                try:
                    shutil.rmtree(tmpdir)
                    break
                except PermissionError:
                    _kill_zombie_postgres_processes()
                    time.sleep(1.0)


@pytest.mark.postgres
class TestPgserverContextManager:
    """Test context manager behavior."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Reset singleton before and after each test."""
        PostgresConnectionManager.reset_instance()
        yield
        PostgresConnectionManager.reset_instance()

    def test_connection_context_commits_on_success(self):
        """Connection context commits on successful exit."""
        tmpdir = tempfile.mkdtemp()
        try:
            config = StorageConfig(
                mode=StorageMode.LOCAL,
                data_dir=str(Path(tmpdir) / "pgdata"),
            )
            manager = PostgresConnectionManager(config)
            manager.start()

            # Create a table and insert data
            with manager.connection("ctx_test") as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, value TEXT)"
                )
                conn.execute("INSERT INTO test_table (value) VALUES ('test')")
                conn.commit()

            # Data should persist
            with manager.connection("ctx_test") as conn:
                result = conn.execute(
                    "SELECT value FROM test_table WHERE value = 'test'"
                ).fetchone()
                assert result is not None
                assert result[0] == "test"

            manager.stop()
            time.sleep(1.0)
        finally:
            # Cleanup with retry
            for _ in range(5):
                try:
                    shutil.rmtree(tmpdir)
                    break
                except PermissionError:
                    _kill_zombie_postgres_processes()
                    time.sleep(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
