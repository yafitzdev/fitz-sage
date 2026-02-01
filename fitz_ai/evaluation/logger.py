# fitz_ai/evaluation/logger.py
"""
Governance Logger - Persist governance decisions for observability.

Logs each governance decision to PostgreSQL for:
- Historical analysis
- Trend monitoring
- Regression detection
- Debugging
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import fitz_ai
from fitz_ai.core.governance import GovernanceDecision, GovernanceLog
from fitz_ai.evaluation.schema import ensure_schema
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import GOVERNANCE

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

    from fitz_ai.core import Chunk

logger = get_logger(__name__)


class GovernanceLogger:
    """
    Logs governance decisions to PostgreSQL and structured logs.

    Thread-safe: uses connection pool for database operations.
    Schema is auto-initialized on first log if needed.

    Usage:
        logger = GovernanceLogger(pool, collection="default")
        log_entry = logger.log(decision, query, chunks)
    """

    def __init__(
        self,
        pool: "ConnectionPool",
        collection: str,
        pipeline_version: str | None = None,
    ):
        """
        Initialize governance logger.

        Args:
            pool: PostgreSQL connection pool
            collection: Collection being logged
            pipeline_version: Version string for tracking regressions
        """
        self.pool = pool
        self.collection = collection
        self.pipeline_version = pipeline_version or fitz_ai.__version__
        self._schema_ensured = False

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
        Create and persist a governance log entry.

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
            chunk_count=len(chunks),
            collection=self.collection,
            latency_ms=latency_ms,
            pipeline_version=self.pipeline_version,
        )

        # Persist to PostgreSQL
        self._persist(log_entry, query)

        # Emit structured log
        logger.info(
            f"{GOVERNANCE} mode={log_entry.mode} "
            f"constraints={log_entry.triggered_constraints} "
            f"chunks={log_entry.chunk_count}",
            extra=log_entry.to_dict(),
        )

        return log_entry

    def _persist(self, log_entry: GovernanceLog, query_text: str) -> None:
        """
        Insert governance log into PostgreSQL.

        Args:
            log_entry: The log entry to persist
            query_text: Original query text for debugging
        """
        self._ensure_schema()

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

        params = (
            log_entry.timestamp,
            log_entry.query_hash,
            query_text,
            log_entry.mode,
            list(log_entry.triggered_constraints),
            list(log_entry.signals),
            list(log_entry.reasons),
            log_entry.chunk_count,
            log_entry.collection,
            log_entry.latency_ms,
            log_entry.pipeline_version,
        )

        try:
            with self.pool.connection() as conn:
                conn.execute(sql, params)
                conn.commit()
        except Exception as e:
            # Log but don't fail the pipeline on logging errors
            logger.warning(f"{GOVERNANCE} Failed to persist log: {e}")


__all__ = ["GovernanceLogger"]
