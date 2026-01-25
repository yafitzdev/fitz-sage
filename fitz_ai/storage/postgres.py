# fitz_ai/storage/postgres.py
"""
PostgreSQL connection management for unified storage.

Handles:
- pgserver lifecycle for local mode (embedded PostgreSQL)
- Connection pooling via psycopg_pool
- Per-collection database creation
- pgvector extension initialization
"""

from __future__ import annotations

import atexit
import re
import threading
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

logger = get_logger(__name__)

# Valid collection name pattern (alphanumeric + underscore)
COLLECTION_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


def _sanitize_collection_name(name: str) -> str:
    """Sanitize collection name for use as database name."""
    # Replace invalid chars with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Ensure starts with letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = "c_" + sanitized
    return sanitized.lower()


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
        self._started = False
        self._initialized_dbs: set[str] = set()

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
                # Register cleanup on exit
                atexit.register(cls._instance.stop)
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

    def _start_pgserver(self, timeout: int = 60) -> None:
        """
        Start embedded pgserver with timeout.

        Args:
            timeout: Maximum seconds to wait for pgserver to start (default: 60)

        Raises:
            TimeoutError: If pgserver doesn't start within timeout
            ImportError: If pgserver package not installed
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

        logger.info(f"{STORAGE} Starting pgserver at {data_dir} (timeout={timeout}s)")

        # Run pgserver startup with timeout to prevent hanging
        import concurrent.futures

        def _get_server():
            return pgserver.get_server(str(data_dir))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_get_server)
            try:
                self._pgserver = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                # Try to clean up
                future.cancel()
                raise TimeoutError(
                    f"pgserver failed to start within {timeout}s. "
                    "Run 'fitz reset' to clear corrupted state."
                )

        self._base_uri = self._pgserver.get_uri()
        logger.info(f"{STORAGE} pgserver started successfully")

    def _get_uri(self, database: str = "postgres") -> str:
        """Get connection URI for a database."""
        if not self._base_uri:
            raise RuntimeError("Connection manager not started. Call start() first.")
        return _replace_database_in_uri(self._base_uri, database)

    def _ensure_database(self, collection: str) -> str:
        """
        Ensure database exists for collection.

        Returns the database name.
        """
        db_name = f"fitz_{_sanitize_collection_name(collection)}"

        if db_name in self._initialized_dbs:
            return db_name

        # Connect to postgres database to create new database
        import psycopg

        postgres_uri = self._get_uri("postgres")

        with psycopg.connect(postgres_uri, autocommit=True) as conn:
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
        with psycopg.connect(db_uri) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            logger.debug(f"{STORAGE} pgvector extension enabled in {db_name}")

        self._initialized_dbs.add(db_name)
        return db_name

    def get_pool(self, collection: str) -> "ConnectionPool":
        """
        Get connection pool for a collection.

        Creates the database if it doesn't exist.

        Args:
            collection: Collection name.

        Returns:
            ConnectionPool for the collection's database.
        """
        if not self._started:
            self.start()

        if collection not in self._pools:
            from psycopg_pool import ConnectionPool

            db_name = self._ensure_database(collection)
            db_uri = self._get_uri(db_name)

            self._pools[collection] = ConnectionPool(
                db_uri,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                open=True,
            )
            logger.debug(f"{STORAGE} Created connection pool for collection '{collection}'")

        return self._pools[collection]

    @contextmanager
    def connection(self, collection: str) -> Generator["Connection", None, None]:
        """
        Get connection for a collection.

        Args:
            collection: Collection name.

        Yields:
            Database connection from pool.
        """
        pool = self.get_pool(collection)
        with pool.connection() as conn:
            yield conn

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

    def stop(self) -> None:
        """Stop PostgreSQL and close all pools."""
        with self._lock:
            # Close all pools
            for name, pool in self._pools.items():
                try:
                    pool.close()
                    logger.debug(f"{STORAGE} Closed pool for '{name}'")
                except Exception as e:
                    logger.warning(f"{STORAGE} Error closing pool '{name}': {e}")

            self._pools.clear()
            self._initialized_dbs.clear()

            # Stop pgserver
            if self._pgserver is not None:
                try:
                    self._pgserver.cleanup()
                    logger.info(f"{STORAGE} pgserver stopped")
                except Exception as e:
                    logger.warning(f"{STORAGE} Error stopping pgserver: {e}")
                self._pgserver = None

            self._base_uri = None
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
