# fitz_ai/evaluation/logger.py
"""
Governance Logger - Persist governance decisions for observability.

Logs each governance decision to PostgreSQL for:
- Historical analysis
- Trend monitoring
- Regression detection
- Debugging

Uses batched inserts to avoid per-query INSERT overhead when running
many queries. Flushes automatically on buffer threshold or explicit flush().
"""

from __future__ import annotations

import atexit
import threading
from typing import TYPE_CHECKING, Sequence

import fitz_ai
from fitz_ai.engines.fitz_rag.governance import GovernanceDecision, GovernanceLog
from fitz_ai.evaluation.schema import ensure_schema
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import GOVERNANCE

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

    from fitz_ai.core import Chunk

logger = get_logger(__name__)

# Default batch size before auto-flush
DEFAULT_BATCH_SIZE = 50


class GovernanceLogger:
    """
    Logs governance decisions to PostgreSQL and structured logs.

    Thread-safe: uses connection pool for database operations and
    thread lock for buffer access.

    Batched inserts: buffers log entries and flushes in batches to
    avoid per-query INSERT overhead. Flushes automatically when:
    - Buffer reaches batch_size entries
    - flush() is called explicitly
    - Instance is garbage collected or process exits

    Usage:
        logger = GovernanceLogger(pool, collection="default")
        log_entry = logger.log(decision, query, chunks)

        # Explicit flush if needed (e.g., before shutdown)
        logger.flush()
    """

    def __init__(
        self,
        pool: "ConnectionPool",
        collection: str,
        pipeline_version: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize governance logger.

        Args:
            pool: PostgreSQL connection pool
            collection: Collection being logged
            pipeline_version: Version string for tracking regressions
            batch_size: Number of entries to buffer before auto-flush
        """
        self.pool = pool
        self.collection = collection
        self.pipeline_version = pipeline_version or fitz_ai.__version__
        self.batch_size = batch_size

        self._buffer: list[tuple[GovernanceLog, str]] = []
        self._lock = threading.Lock()
        self._schema_ensured = False

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Flush remaining entries on exit."""
        try:
            self.flush()
        except Exception as e:
            logger.debug(f"{GOVERNANCE} Cleanup flush failed: {e}")

    def _ensure_schema(self) -> None:
        """Ensure schema exists (lazy, one-time)."""
        if self._schema_ensured:
            return
        with self.pool.connection() as conn:
            ensure_schema(conn)
        self._schema_ensured = True

    def log(
        self,
        decision: GovernanceDecision,
        query: str,
        chunks: Sequence["Chunk"],
        latency_ms: float | None = None,
    ) -> GovernanceLog:
        """
        Create and buffer a governance log entry.

        The entry is added to a buffer and persisted in batches.
        Call flush() to force immediate persistence if needed.

        Args:
            decision: The governance decision to log
            query: The original query string
            chunks: Retrieved chunks available for the decision
            latency_ms: Time to make governance decision in milliseconds

        Returns:
            The created GovernanceLog entry
        """
        # Create log entry
        query_hash = GovernanceLog.hash_query(query)
        log_entry = GovernanceLog.from_decision(
            decision,
            query_hash=query_hash,
            query_text=query,
            chunk_count=len(chunks),
            collection=self.collection,
            latency_ms=latency_ms,
            pipeline_version=self.pipeline_version,
        )

        # Add to buffer (thread-safe)
        should_flush = False
        with self._lock:
            self._buffer.append((log_entry, query))
            if len(self._buffer) >= self.batch_size:
                should_flush = True

        # Flush outside lock if needed
        if should_flush:
            self.flush()

        # Emit structured log (always immediate for observability)
        logger.info(
            f"{GOVERNANCE} mode={log_entry.mode} "
            f"constraints={log_entry.triggered_constraints} "
            f"chunks={log_entry.chunk_count}",
            extra=log_entry.to_dict(),
        )

        return log_entry

    def flush(self) -> int:
        """
        Flush all buffered entries to PostgreSQL.

        Uses a single transaction with multi-row INSERT for efficiency.

        Returns:
            Number of entries flushed
        """
        # Grab buffer atomically
        with self._lock:
            if not self._buffer:
                return 0
            entries = self._buffer[:]
            self._buffer.clear()

        # Persist batch
        try:
            self._persist_batch(entries)
            logger.debug(f"{GOVERNANCE} Flushed {len(entries)} log entries")
            return len(entries)
        except Exception as e:
            # On failure, put entries back for retry
            logger.warning(f"{GOVERNANCE} Batch flush failed: {e}")
            with self._lock:
                self._buffer = entries + self._buffer
            return 0

    def _persist_batch(self, entries: list[tuple[GovernanceLog, str]]) -> None:
        """
        Insert multiple governance logs in a single transaction.

        Uses execute_values for efficient multi-row INSERT.

        Args:
            entries: List of (GovernanceLog, query_text) tuples
        """
        if not entries:
            return

        self._ensure_schema()

        # Build multi-row INSERT using VALUES clause
        # psycopg3 handles this efficiently with executemany
        sql = """
            INSERT INTO governance_logs (
                timestamp,
                query_hash,
                query_text,
                mode,
                triggered_constraints,
                signals,
                reasons,
                chunk_count,
                collection,
                latency_ms,
                pipeline_version
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """

        params_list = [
            (
                log_entry.timestamp,
                log_entry.query_hash,
                log_entry.query_text,
                log_entry.mode,
                list(log_entry.triggered_constraints),
                list(log_entry.signals),
                list(log_entry.reasons),
                log_entry.chunk_count,
                log_entry.collection,
                log_entry.latency_ms,
                log_entry.pipeline_version,
            )
            for log_entry, _query_text in entries
        ]

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # Use executemany for batched execution
                cur.executemany(sql, params_list)
            conn.commit()

    def pending_count(self) -> int:
        """Get number of entries waiting to be flushed."""
        with self._lock:
            return len(self._buffer)


__all__ = ["GovernanceLogger"]
