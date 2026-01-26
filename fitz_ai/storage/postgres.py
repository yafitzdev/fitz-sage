# fitz_ai/storage/postgres.py
"""
PostgreSQL connection management for unified storage.

Handles:
- pgserver lifecycle for local mode (embedded PostgreSQL)
- Connection pooling via psycopg_pool with health checks
- Per-collection database creation
- pgvector extension initialization
- Graceful shutdown on signals (SIGTERM, SIGINT)
- Auto-recovery from corrupted pgdata and stale connections
- Automatic restart with exponential backoff on failures
- Lock file cleanup for crashed instances
"""

from __future__ import annotations

import atexit
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Optional
from urllib.parse import urlparse, urlunparse

from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import STORAGE
from fitz_ai.storage.config import StorageConfig, StorageMode

if TYPE_CHECKING:
    from psycopg import Connection
    from psycopg_pool import ConnectionPool

# Lazy imports for psycopg (done once at module level when first needed)
_psycopg = None
_psycopg_pool = None


def _get_psycopg():
    """Lazy import psycopg once."""
    global _psycopg
    if _psycopg is None:
        import psycopg
        _psycopg = psycopg
    return _psycopg


def _get_psycopg_pool():
    """Lazy import psycopg_pool once."""
    global _psycopg_pool
    if _psycopg_pool is None:
        from psycopg_pool import ConnectionPool
        _psycopg_pool = ConnectionPool
    return _psycopg_pool


logger = get_logger(__name__)

# Track if signal handlers have been registered
_signal_handlers_registered = False
_original_sigterm_handler = None
_original_sigint_handler = None


def _graceful_shutdown_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals by stopping pgserver gracefully."""
    logger.info(f"{STORAGE} Received signal {signum}, shutting down pgserver gracefully...")

    # Stop the connection manager
    if PostgresConnectionManager._instance is not None:
        PostgresConnectionManager._instance.stop()

    # Call original handler if it exists
    original = _original_sigterm_handler if signum == signal.SIGTERM else _original_sigint_handler
    if original is not None and callable(original):
        original(signum, frame)
    elif original == signal.SIG_DFL:
        # Default behavior - re-raise signal
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)


def _register_signal_handlers() -> None:
    """Register signal handlers for graceful shutdown."""
    global _signal_handlers_registered, _original_sigterm_handler, _original_sigint_handler

    if _signal_handlers_registered:
        return

    # Only register in main thread
    if threading.current_thread() is not threading.main_thread():
        return

    try:
        _original_sigterm_handler = signal.signal(signal.SIGTERM, _graceful_shutdown_handler)
        _original_sigint_handler = signal.signal(signal.SIGINT, _graceful_shutdown_handler)
        _signal_handlers_registered = True
        logger.debug(f"{STORAGE} Registered signal handlers for graceful shutdown")
    except (ValueError, OSError) as e:
        # Signal handling not available (e.g., not main thread, or Windows limitations)
        logger.debug(f"{STORAGE} Could not register signal handlers: {e}")

# Valid collection name pattern (alphanumeric + underscore)
COLLECTION_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

# Retry configuration
MAX_RESTART_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 0.5  # seconds
RETRY_BACKOFF_MAX = 10.0  # seconds


def _kill_zombie_postgres_processes() -> None:
    """
    Kill any zombie postgres processes that might hold locks on pgdata.

    This is a best-effort cleanup for cases where postgres crashed or was
    killed without proper shutdown, leaving lock files behind.
    """
    try:
        if sys.platform == "win32":
            # Windows: kill postgres.exe processes
            result = subprocess.run(
                ["taskkill", "/F", "/IM", "postgres.exe"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.debug(f"{STORAGE} Killed zombie postgres processes")
        else:
            # Unix: kill postgres processes owned by current user
            result = subprocess.run(
                ["pkill", "-9", "-u", str(subprocess.os.getuid()), "postgres"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.debug(f"{STORAGE} Killed zombie postgres processes")
    except Exception as e:
        logger.debug(f"{STORAGE} Could not kill postgres processes (may not exist): {e}")


def _calculate_backoff(attempt: int) -> float:
    """Calculate exponential backoff delay."""
    delay = RETRY_BACKOFF_BASE * (2 ** attempt)
    return min(delay, RETRY_BACKOFF_MAX)


def _cleanup_stale_lock_files(data_dir: Path) -> None:
    """
    Remove stale PostgreSQL lock files that prevent startup.

    PostgreSQL creates postmaster.pid and socket lock files that can
    persist after crashes, preventing new instances from starting.
    """
    lock_files = [
        data_dir / "postmaster.pid",
        data_dir / ".s.PGSQL.5432.lock",
    ]

    # Also clean up any socket files
    for item in data_dir.glob(".s.PGSQL.*"):
        lock_files.append(item)

    for lock_file in lock_files:
        if lock_file.exists():
            try:
                lock_file.unlink()
                logger.debug(f"{STORAGE} Removed stale lock file: {lock_file.name}")
            except Exception as e:
                logger.warning(f"{STORAGE} Could not remove lock file {lock_file}: {e}")


def _force_remove_pgdata(data_dir: Path, max_retries: int = 3) -> bool:
    """
    Forcefully remove pgdata directory, killing processes if needed.

    Args:
        data_dir: Path to pgdata directory
        max_retries: Maximum number of retry attempts

    Returns:
        True if successfully removed, False otherwise
    """
    for attempt in range(max_retries):
        try:
            if data_dir.exists():
                shutil.rmtree(data_dir)
            return True
        except PermissionError as e:
            logger.warning(
                f"{STORAGE} Cannot remove pgdata (attempt {attempt + 1}/{max_retries}): {e}"
            )
            # Try killing postgres processes that might hold locks
            _kill_zombie_postgres_processes()
            time.sleep(1)  # Give OS time to release file handles
        except Exception as e:
            logger.error(f"{STORAGE} Unexpected error removing pgdata: {e}")
            return False

    return False


# Cache for sanitized collection names (avoid regex on hot path)
_sanitized_name_cache: dict[str, str] = {}


def _sanitize_collection_name(name: str) -> str:
    """Sanitize collection name for use as database name (cached)."""
    if name in _sanitized_name_cache:
        return _sanitized_name_cache[name]

    # Replace invalid chars with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Ensure starts with letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = "c_" + sanitized
    result = sanitized.lower()

    _sanitized_name_cache[name] = result
    return result


def _replace_database_in_uri(uri: str, database: str) -> str:
    """Replace database name in PostgreSQL connection URI."""
    parsed = urlparse(uri)
    # Path is /dbname, replace it
    new_path = f"/{database}"
    return urlunparse(parsed._replace(path=new_path))


class PostgresConnectionManager:
    """
    Manages PostgreSQL connections for fitz-ai.

    Singleton that handles:
    - pgserver lifecycle for local mode
    - Connection pooling per collection
    - Database creation per collection
    - pgvector extension initialization
    """

    _instance: Optional["PostgresConnectionManager"] = None
    _lock = threading.Lock()

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize connection manager.

        Args:
            config: Storage configuration. Uses defaults if None.
        """
        self.config = config or StorageConfig()
        self._pgserver: Any = None  # pgserver.Server instance
        self._pools: dict[str, "ConnectionPool"] = {}  # collection -> pool
        self._base_uri: Optional[str] = None
        self._uri_cache: dict[str, str] = {}  # database -> URI cache
        self._started = False
        self._initialized_dbs: set[str] = set()
        self._restart_attempts = 0  # Track restart attempts to prevent infinite loops
        self._last_health_check: float = 0.0  # Timestamp of last health check
        self._health_check_interval = 30.0  # Seconds between proactive health checks

    @classmethod
    def get_instance(cls, config: Optional[StorageConfig] = None) -> "PostgresConnectionManager":
        """
        Get or create singleton instance.

        Args:
            config: Storage configuration. Only used on first call.

        Returns:
            The singleton PostgresConnectionManager instance.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
                # Register cleanup on exit and signal handlers
                atexit.register(cls._instance.stop)
                _register_signal_handlers()
            elif config is not None and not cls._instance._started:
                # Allow config update before start
                cls._instance.config = config
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None

    def start(self) -> None:
        """Start PostgreSQL server (pgserver for local mode)."""
        if self._started:
            return

        with self._lock:
            if self._started:
                return

            if self.config.mode == StorageMode.LOCAL:
                self._start_pgserver()
            else:
                self._base_uri = self.config.connection_string
                if not self._base_uri:
                    raise ValueError("connection_string required for external mode")

            self._started = True
            logger.info(f"{STORAGE} PostgreSQL connection manager started (mode={self.config.mode.value})")

    def _start_pgserver(self, timeout: int = 60, allow_recovery: bool = True) -> None:
        """
        Start embedded pgserver with timeout and auto-recovery.

        Args:
            timeout: Maximum seconds to wait for pgserver to start (default: 60)
            allow_recovery: If True, attempt recovery by resetting pgdata on failure

        Raises:
            TimeoutError: If pgserver doesn't start within timeout
            ImportError: If pgserver package not installed
            RuntimeError: If pgserver fails to start even after recovery attempt
        """
        try:
            import pgserver
        except ImportError as e:
            raise ImportError(
                "pgserver not installed. Install with: pip install pgserver"
            ) from e

        data_dir = self.config.data_dir or FitzPaths.ensure_pgdata()
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Clean up any stale lock files from previous crashes
        _cleanup_stale_lock_files(data_dir)

        logger.info(f"{STORAGE} Starting pgserver at {data_dir} (timeout={timeout}s)")

        # Run pgserver startup with timeout to prevent hanging
        import concurrent.futures

        def _get_server():
            return pgserver.get_server(str(data_dir))

        def _try_start() -> bool:
            """Attempt to start pgserver. Returns True on success."""
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_get_server)
                try:
                    self._pgserver = future.result(timeout=timeout)
                    self._base_uri = self._pgserver.get_uri()
                    return True
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    return False
                except Exception as e:
                    logger.warning(f"{STORAGE} pgserver startup failed: {e}")
                    return False

        # First attempt
        if _try_start():
            logger.info(f"{STORAGE} pgserver started successfully")
            return

        # Recovery attempt - reset pgdata and try again
        if allow_recovery:
            logger.warning(f"{STORAGE} pgserver failed to start, attempting auto-recovery...")
            try:
                # Clean up any partial state
                if self._pgserver is not None:
                    try:
                        self._pgserver.cleanup()
                    except Exception:
                        pass
                    self._pgserver = None

                # Kill any zombie postgres processes that might hold locks
                _kill_zombie_postgres_processes()
                time.sleep(0.5)  # Give OS time to release handles

                # Delete corrupted pgdata with retries
                if data_dir.exists():
                    if _force_remove_pgdata(data_dir):
                        logger.info(f"{STORAGE} Deleted corrupted pgdata at {data_dir}")
                    else:
                        logger.error(
                            f"{STORAGE} Could not delete pgdata at {data_dir}. "
                            "Try manually deleting it or running 'fitz reset'."
                        )
                        raise RuntimeError(f"Cannot remove corrupted pgdata at {data_dir}")

                # Recreate directory
                data_dir.mkdir(parents=True, exist_ok=True)

                # Retry without recovery (to prevent infinite loop)
                if _try_start():
                    logger.info(f"{STORAGE} pgserver started successfully after recovery")
                    return
            except Exception as e:
                logger.error(f"{STORAGE} Auto-recovery failed: {e}")

        # All attempts failed
        raise RuntimeError(
            f"pgserver failed to start within {timeout}s even after recovery attempt. "
            "Try manually running 'fitz reset' or deleting ~/.fitz/pgdata"
        )

    def _get_uri(self, database: str = "postgres") -> str:
        """Get connection URI for a database (cached)."""
        if not self._base_uri:
            raise RuntimeError("Connection manager not started. Call start() first.")

        # Check cache first
        if database in self._uri_cache:
            return self._uri_cache[database]

        uri = _replace_database_in_uri(self._base_uri, database)
        self._uri_cache[database] = uri
        return uri

    def _ensure_database(self, collection: str, _retry: bool = True) -> str:
        """
        Ensure database exists for collection.

        Auto-recovers from connection failures by restarting pgserver.
        Uses caching - only checks database existence once per session.

        Returns the database name.
        """
        psycopg = _get_psycopg()

        db_name = f"fitz_{_sanitize_collection_name(collection)}"

        # Fast path: already initialized this session
        if db_name in self._initialized_dbs:
            return db_name

        postgres_uri = self._get_uri("postgres")

        try:
            with psycopg.connect(postgres_uri, autocommit=True, connect_timeout=30) as conn:
                # Check if database exists
                result = conn.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
                ).fetchone()

                if not result:
                    # Create database
                    conn.execute(f'CREATE DATABASE "{db_name}"')
                    logger.info(f"{STORAGE} Created database: {db_name}")

            # Initialize pgvector extension in new database
            db_uri = self._get_uri(db_name)
            with psycopg.connect(db_uri, connect_timeout=30) as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                logger.debug(f"{STORAGE} pgvector extension enabled in {db_name}")

            self._initialized_dbs.add(db_name)
            return db_name

        except (psycopg.OperationalError, psycopg.errors.ConnectionTimeout) as e:
            if not _retry:
                raise

            # Auto-recovery: restart pgserver and retry
            logger.warning(f"{STORAGE} Connection failed ({e}), attempting auto-recovery...")
            self._restart_pgserver()
            return self._ensure_database(collection, _retry=False)

    def _restart_pgserver(self) -> None:
        """Restart pgserver after connection failure with backoff."""
        self._restart_attempts += 1

        if self._restart_attempts > MAX_RESTART_ATTEMPTS:
            raise RuntimeError(
                f"pgserver restart failed after {MAX_RESTART_ATTEMPTS} attempts. "
                "Try manually running 'fitz reset' or deleting ~/.fitz/pgdata"
            )

        backoff = _calculate_backoff(self._restart_attempts - 1)
        logger.warning(
            f"{STORAGE} Restart attempt {self._restart_attempts}/{MAX_RESTART_ATTEMPTS} "
            f"(backoff: {backoff:.1f}s)"
        )

        with self._lock:
            # Close all pools
            for name, pool in list(self._pools.items()):
                try:
                    pool.close()
                except Exception:
                    pass
            self._pools.clear()
            self._initialized_dbs.clear()
            self._uri_cache.clear()  # Clear URI cache

            # Stop pgserver
            if self._pgserver is not None:
                try:
                    self._pgserver.cleanup()
                except Exception:
                    pass
                self._pgserver = None

            self._base_uri = None
            self._started = False

            # Kill any zombie processes
            _kill_zombie_postgres_processes()

            # Exponential backoff before restart
            time.sleep(backoff)

            # Clean up lock files
            data_dir = self.config.data_dir or FitzPaths.ensure_pgdata()
            _cleanup_stale_lock_files(Path(data_dir))

            # Restart
            logger.info(f"{STORAGE} Restarting pgserver...")
            self._start_pgserver()
            self._started = True

            # Reset counter on successful restart
            self._restart_attempts = 0
            logger.info(f"{STORAGE} pgserver restarted successfully")

    def get_pool(self, collection: str) -> "ConnectionPool":
        """
        Get connection pool for a collection.

        Creates the database if it doesn't exist. Pool has built-in health
        checking via check= parameter - no manual verification needed.

        Args:
            collection: Collection name.

        Returns:
            ConnectionPool for the collection's database.
        """
        if not self._started:
            self.start()

        if collection not in self._pools:
            ConnectionPool = _get_psycopg_pool()

            db_name = self._ensure_database(collection)
            db_uri = self._get_uri(db_name)

            # Configure pool with built-in health checks
            # check= verifies connections before returning them
            # reconnect_timeout= handles stale connections automatically
            self._pools[collection] = ConnectionPool(
                db_uri,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                open=True,
                check=ConnectionPool.check_connection,
                reconnect_timeout=30.0,
                timeout=30.0,
            )
            logger.debug(f"{STORAGE} Created connection pool for collection '{collection}'")

        return self._pools[collection]

    @contextmanager
    def connection(self, collection: str) -> Generator["Connection", None, None]:
        """
        Get connection for a collection.

        The pool handles health checking and reconnection automatically.
        Only falls back to pgserver restart for catastrophic failures.

        Args:
            collection: Collection name.

        Yields:
            Database connection from pool.
        """
        psycopg = _get_psycopg()

        try:
            pool = self.get_pool(collection)
            with pool.connection() as conn:
                yield conn
        except (psycopg.OperationalError, psycopg.errors.ConnectionTimeout) as e:
            # Pool's reconnect failed - try pgserver restart as last resort
            logger.warning(f"{STORAGE} Pool connection failed ({e}), attempting recovery...")

            # Clear this pool
            if collection in self._pools:
                try:
                    self._pools[collection].close()
                except Exception:
                    pass
                del self._pools[collection]

            # Try restarting pgserver
            try:
                self._restart_pgserver()
                # Retry once with fresh pool
                pool = self.get_pool(collection)
                with pool.connection() as conn:
                    yield conn
            except Exception as retry_error:
                raise RuntimeError(
                    f"Connection failed even after pgserver restart: {retry_error}"
                ) from e

    def execute(self, collection: str, sql: str, params: tuple = ()) -> Any:
        """
        Execute SQL on a collection's database.

        Args:
            collection: Collection name.
            sql: SQL statement.
            params: Query parameters.

        Returns:
            Cursor result.
        """
        with self.connection(collection) as conn:
            result = conn.execute(sql, params)
            conn.commit()
            return result

    def is_healthy(self) -> tuple[bool, str]:
        """
        Check if PostgreSQL connection is healthy.

        Returns:
            Tuple of (is_healthy, message)
        """
        if not self._started:
            return False, "Not started"

        try:
            psycopg = _get_psycopg()
            postgres_uri = self._get_uri("postgres")
            with psycopg.connect(postgres_uri, autocommit=True, connect_timeout=5) as conn:
                result = conn.execute("SELECT version()").fetchone()
                version = result[0] if result else "unknown"
                return True, f"Connected to {version[:50]}..."
        except Exception as e:
            return False, f"Connection failed: {e}"

    def get_stats(self) -> dict[str, Any]:
        """
        Get connection manager statistics.

        Returns:
            Dict with stats about pools, connections, and restarts.
        """
        stats = {
            "started": self._started,
            "mode": self.config.mode.value if self.config else "unknown",
            "pools": len(self._pools),
            "databases": len(self._initialized_dbs),
            "restart_attempts": self._restart_attempts,
        }

        # Add pool-level stats
        pool_stats = {}
        for name, pool in self._pools.items():
            try:
                pool_stats[name] = {
                    "size": pool.get_stats().get("pool_size", 0),
                    "available": pool.get_stats().get("pool_available", 0),
                }
            except Exception:
                pool_stats[name] = {"error": "Could not get stats"}

        stats["pool_details"] = pool_stats
        return stats

    def stop(self) -> None:
        """Stop PostgreSQL and close all pools.

        This method is defensive and handles partially-initialized instances
        (e.g., from crashes during __init__ or interrupted startup).
        """
        with self._lock:
            # Close all pools (defensive - check attribute exists)
            if hasattr(self, "_pools") and self._pools:
                for name, pool in list(self._pools.items()):
                    try:
                        pool.close()
                        logger.debug(f"{STORAGE} Closed pool for '{name}'")
                    except Exception as e:
                        logger.warning(f"{STORAGE} Error closing pool '{name}': {e}")
                self._pools.clear()

            if hasattr(self, "_initialized_dbs"):
                self._initialized_dbs.clear()

            if hasattr(self, "_uri_cache"):
                self._uri_cache.clear()

            # Stop pgserver
            if hasattr(self, "_pgserver") and self._pgserver is not None:
                try:
                    self._pgserver.cleanup()
                    logger.info(f"{STORAGE} pgserver stopped")
                except Exception as e:
                    logger.warning(f"{STORAGE} Error stopping pgserver: {e}")
                self._pgserver = None

            if hasattr(self, "_base_uri"):
                self._base_uri = None
            if hasattr(self, "_started"):
                self._started = False


# Module-level convenience functions


def get_connection_manager(config: Optional[StorageConfig] = None) -> PostgresConnectionManager:
    """Get the singleton connection manager."""
    return PostgresConnectionManager.get_instance(config)


@contextmanager
def get_connection(collection: str) -> Generator["Connection", None, None]:
    """
    Get database connection for a collection.

    Convenience function that uses the singleton manager.

    Args:
        collection: Collection name.

    Yields:
        Database connection.
    """
    manager = get_connection_manager()
    with manager.connection(collection) as conn:
        yield conn
