# fitz_ai/evaluation/stats.py
"""
Governance Statistics - Query and aggregate governance data.

Provides methods for:
- Mode distribution over time periods
- Constraint trigger frequency
- Flip detection (behavioral changes)
- Trend analysis for monitoring
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal

from fitz_ai.evaluation.models import (
    AbstainTrend,
    ConstraintFrequency,
    GovernanceFlip,
    ModeDistribution,
)
from fitz_ai.evaluation.schema import ensure_schema
from fitz_ai.logging.logger import get_logger

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

logger = get_logger(__name__)

TimeBucket = Literal["hour", "day", "week"]


class GovernanceStats:
    """
    Query and aggregate governance statistics.

    All methods are read-only and safe to call concurrently.
    Schema is auto-ensured on first query if needed.

    Usage:
        stats = GovernanceStats(pool)
        dist = stats.get_mode_distribution(collection="default", days=7)
        print(f"Abstain rate: {dist.abstain_rate:.1%}")
    """

    def __init__(self, pool: "ConnectionPool"):
        """
        Initialize stats aggregator.

        Args:
            pool: PostgreSQL connection pool
        """
        self.pool = pool
        self._schema_ensured = False

    def _ensure_schema(self) -> None:
        """Ensure schema exists (lazy, one-time)."""
        if self._schema_ensured:
            return
        with self.pool.connection() as conn:
            ensure_schema(conn)
        self._schema_ensured = True

    def get_mode_distribution(
        self,
        collection: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        days: int | None = None,
    ) -> ModeDistribution:
        """
        Get mode distribution for a time period.

        Args:
            collection: Filter by collection (None = all collections)
            start: Start of period (UTC). Defaults to 7 days ago.
            end: End of period (UTC). Defaults to now.
            days: Shorthand for start = now - days (overrides start if provided)

        Returns:
            ModeDistribution with counts and rates
        """
        self._ensure_schema()

        # Resolve time range
        now = datetime.now(timezone.utc)
        if days is not None:
            start = now - timedelta(days=days)
        if start is None:
            start = now - timedelta(days=7)
        if end is None:
            end = now

        # Build query
        sql = """
            SELECT
                mode,
                COUNT(*) as count
            FROM governance_logs
            WHERE timestamp >= %s AND timestamp <= %s
        """
        params: list = [start, end]

        if collection is not None:
            sql += " AND collection = %s"
            params.append(collection)

        sql += " GROUP BY mode"

        # Execute
        with self.pool.connection() as conn:
            result = conn.execute(sql, tuple(params))
            rows = result.fetchall()

        # Aggregate
        counts = {"confident": 0, "qualified": 0, "disputed": 0, "abstain": 0}
        for mode, count in rows:
            if mode in counts:
                counts[mode] = count

        total = sum(counts.values())

        return ModeDistribution(
            period_start=start,
            period_end=end,
            total_queries=total,
            confident_count=counts["confident"],
            qualified_count=counts["qualified"],
            disputed_count=counts["disputed"],
            abstain_count=counts["abstain"],
        )

    def get_constraint_frequency(
        self,
        collection: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        days: int | None = None,
        top_n: int = 10,
    ) -> list[ConstraintFrequency]:
        """
        Get most frequently triggered constraints.

        Args:
            collection: Filter by collection (None = all)
            start: Start of period (UTC)
            end: End of period (UTC)
            days: Shorthand for start = now - days
            top_n: Number of top constraints to return

        Returns:
            List of ConstraintFrequency sorted by trigger count descending
        """
        self._ensure_schema()

        # Resolve time range
        now = datetime.now(timezone.utc)
        if days is not None:
            start = now - timedelta(days=days)
        if start is None:
            start = now - timedelta(days=7)
        if end is None:
            end = now

        # Build query - unnest the array to count individual constraints
        sql = """
            WITH constraints_unnested AS (
                SELECT unnest(triggered_constraints) as constraint_name
                FROM governance_logs
                WHERE timestamp >= %s AND timestamp <= %s
        """
        params: list = [start, end]

        if collection is not None:
            sql += " AND collection = %s"
            params.append(collection)

        sql += """
            ),
            total_queries AS (
                SELECT COUNT(*) as total
                FROM governance_logs
                WHERE timestamp >= %s AND timestamp <= %s
        """
        params.extend([start, end])

        if collection is not None:
            sql += " AND collection = %s"
            params.append(collection)

        sql += """
            )
            SELECT
                c.constraint_name,
                COUNT(*) as trigger_count,
                t.total as total_queries
            FROM constraints_unnested c, total_queries t
            GROUP BY c.constraint_name, t.total
            ORDER BY trigger_count DESC
            LIMIT %s
        """
        params.append(top_n)

        # Execute
        with self.pool.connection() as conn:
            result = conn.execute(sql, tuple(params))
            rows = result.fetchall()

        return [
            ConstraintFrequency(
                constraint_name=name,
                trigger_count=count,
                total_queries=total,
            )
            for name, count, total in rows
        ]

    def detect_flips(
        self,
        since: datetime | None = None,
        days: int | None = None,
    ) -> list[GovernanceFlip]:
        """
        Find queries that changed mode over time.

        Detects behavioral changes by comparing the most recent decision
        for each query hash with its previous decision. Includes pipeline
        versions to help identify which changes caused regressions.

        Args:
            since: Only consider decisions after this time
            days: Shorthand for since = now - days

        Returns:
            List of GovernanceFlip sorted by new_timestamp descending
        """
        self._ensure_schema()

        # Resolve time range
        now = datetime.now(timezone.utc)
        if days is not None:
            since = now - timedelta(days=days)
        if since is None:
            since = now - timedelta(days=30)  # Default 30 days

        # Find queries with multiple decisions that changed mode
        # Include pipeline_version for regression tracking
        sql = """
            WITH ranked AS (
                SELECT
                    query_hash,
                    query_text,
                    mode,
                    timestamp,
                    pipeline_version,
                    ROW_NUMBER() OVER (PARTITION BY query_hash ORDER BY timestamp DESC) as rn
                FROM governance_logs
                WHERE timestamp >= %s
            ),
            current_and_previous AS (
                SELECT
                    r1.query_hash,
                    r1.query_text,
                    r1.mode as new_mode,
                    r1.timestamp as new_timestamp,
                    r1.pipeline_version as new_version,
                    r2.mode as old_mode,
                    r2.timestamp as old_timestamp,
                    r2.pipeline_version as old_version
                FROM ranked r1
                JOIN ranked r2 ON r1.query_hash = r2.query_hash AND r2.rn = 2
                WHERE r1.rn = 1 AND r1.mode != r2.mode
            )
            SELECT
                query_hash,
                query_text,
                old_mode,
                new_mode,
                old_timestamp,
                new_timestamp,
                old_version,
                new_version
            FROM current_and_previous
            ORDER BY new_timestamp DESC
        """

        with self.pool.connection() as conn:
            result = conn.execute(sql, (since,))
            rows = result.fetchall()

        return [
            GovernanceFlip(
                query_hash=row[0],
                query_text=row[1],
                old_mode=row[2],
                new_mode=row[3],
                old_timestamp=row[4],
                new_timestamp=row[5],
                old_version=row[6],
                new_version=row[7],
            )
            for row in rows
        ]

    def get_abstain_rate_trend(
        self,
        collection: str | None = None,
        bucket: TimeBucket = "day",
        days: int = 30,
    ) -> list[AbstainTrend]:
        """
        Get abstain rate over time for alerting.

        Args:
            collection: Filter by collection (None = all)
            bucket: Time bucket size ("hour", "day", "week")
            days: Number of days to look back

        Returns:
            List of AbstainTrend sorted by bucket_start ascending
        """
        self._ensure_schema()

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days)

        # PostgreSQL date_trunc for bucketing
        bucket_expr = f"date_trunc('{bucket}', timestamp)"

        sql = f"""
            SELECT
                {bucket_expr} as bucket_start,
                COUNT(*) FILTER (WHERE mode = 'abstain') as abstain_count,
                COUNT(*) as total_count
            FROM governance_logs
            WHERE timestamp >= %s
        """
        params: list = [start]

        if collection is not None:
            sql += " AND collection = %s"
            params.append(collection)

        sql += f"""
            GROUP BY {bucket_expr}
            ORDER BY bucket_start ASC
        """

        with self.pool.connection() as conn:
            result = conn.execute(sql, tuple(params))
            rows = result.fetchall()

        return [
            AbstainTrend(
                bucket_start=row[0],
                abstain_rate=row[1] / row[2] if row[2] > 0 else 0.0,
                total_queries=row[2],
            )
            for row in rows
        ]

    def get_total_queries(
        self,
        collection: str | None = None,
        days: int | None = None,
    ) -> int:
        """
        Get total number of logged queries.

        Args:
            collection: Filter by collection (None = all)
            days: Limit to last N days (None = all time)

        Returns:
            Total query count
        """
        self._ensure_schema()

        sql = "SELECT COUNT(*) FROM governance_logs WHERE 1=1"
        params: list = []

        if days is not None:
            start = datetime.now(timezone.utc) - timedelta(days=days)
            sql += " AND timestamp >= %s"
            params.append(start)

        if collection is not None:
            sql += " AND collection = %s"
            params.append(collection)

        with self.pool.connection() as conn:
            result = conn.execute(sql, tuple(params))
            row = result.fetchone()

        return row[0] if row else 0


__all__ = ["GovernanceStats"]
