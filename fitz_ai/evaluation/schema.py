# fitz_ai/evaluation/schema.py
"""
PostgreSQL schema for governance observability.

Provides DDL for the governance_logs table and utility functions
for schema management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fitz_ai.logging.logger import get_logger

if TYPE_CHECKING:
    from psycopg import Connection
    from psycopg_pool import ConnectionPool

logger = get_logger(__name__)


# =============================================================================
# Schema DDL
# =============================================================================

GOVERNANCE_LOGS_TABLE = """
CREATE TABLE IF NOT EXISTS governance_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query_hash VARCHAR(64) NOT NULL,
    query_text TEXT,
    mode VARCHAR(20) NOT NULL,
    triggered_constraints TEXT[],
    signals TEXT[],
    reasons TEXT[],
    chunk_count INT,
    collection VARCHAR(255),
    latency_ms REAL,
    pipeline_version VARCHAR(50)
);
"""

GOVERNANCE_LOGS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_gov_timestamp ON governance_logs(timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_gov_mode ON governance_logs(mode);",
    "CREATE INDEX IF NOT EXISTS idx_gov_query_hash ON governance_logs(query_hash);",
    "CREATE INDEX IF NOT EXISTS idx_gov_collection ON governance_logs(collection);",
]


# =============================================================================
# Schema Management
# =============================================================================


def ensure_schema(conn: "Connection") -> None:
    """
    Ensure governance_logs table and indexes exist.

    Safe to call multiple times - uses IF NOT EXISTS.

    Args:
        conn: PostgreSQL connection
    """
    # Create table
    conn.execute(GOVERNANCE_LOGS_TABLE)

    # Create indexes
    for index_sql in GOVERNANCE_LOGS_INDEXES:
        conn.execute(index_sql)

    conn.commit()
    logger.debug("[GOVERNANCE] Schema ensured for governance_logs table")


def ensure_schema_from_pool(pool: "ConnectionPool") -> None:
    """
    Ensure schema using a connection from the pool.

    Args:
        pool: PostgreSQL connection pool
    """
    with pool.connection() as conn:
        ensure_schema(conn)


def drop_schema(conn: "Connection") -> None:
    """
    Drop governance_logs table (for testing).

    Args:
        conn: PostgreSQL connection
    """
    conn.execute("DROP TABLE IF EXISTS governance_logs CASCADE;")
    conn.commit()
    logger.debug("[GOVERNANCE] Dropped governance_logs table")


__all__ = [
    "GOVERNANCE_LOGS_TABLE",
    "GOVERNANCE_LOGS_INDEXES",
    "ensure_schema",
    "ensure_schema_from_pool",
    "drop_schema",
]
