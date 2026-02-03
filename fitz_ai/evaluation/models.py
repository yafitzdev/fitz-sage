# fitz_ai/evaluation/models.py
"""
Governance observability models.

Data structures for tracking, aggregating, and analyzing governance decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ModeDistribution:
    """
    Distribution of answer modes over a time period.

    Aggregates governance decisions to understand system behavior trends.
    Used by CLI commands and monitoring dashboards.

    Attributes:
        period_start: Start of the aggregation period (UTC)
        period_end: End of the aggregation period (UTC)
        total_queries: Total number of governance decisions in period
        confident_count: Queries that resulted in CONFIDENT mode
        qualified_count: Queries that resulted in QUALIFIED mode
        disputed_count: Queries that resulted in DISPUTED mode
        abstain_count: Queries that resulted in ABSTAIN mode
    """

    period_start: datetime
    period_end: datetime
    total_queries: int
    confident_count: int
    qualified_count: int
    disputed_count: int
    abstain_count: int

    @property
    def abstain_rate(self) -> float:
        """
        Rate of queries that resulted in ABSTAIN.

        High abstain rate may indicate:
        - Insufficient corpus coverage
        - Overly strict constraints
        - Queries outside domain
        """
        if self.total_queries == 0:
            return 0.0
        return self.abstain_count / self.total_queries

    @property
    def confident_rate(self) -> float:
        """
        Rate of queries that resulted in CONFIDENT.

        Healthy systems should have high confident rate for
        in-domain queries with sufficient evidence.
        """
        if self.total_queries == 0:
            return 0.0
        return self.confident_count / self.total_queries

    @property
    def qualified_rate(self) -> float:
        """Rate of queries that resulted in QUALIFIED."""
        if self.total_queries == 0:
            return 0.0
        return self.qualified_count / self.total_queries

    @property
    def disputed_rate(self) -> float:
        """Rate of queries that resulted in DISPUTED."""
        if self.total_queries == 0:
            return 0.0
        return self.disputed_count / self.total_queries

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_queries": self.total_queries,
            "confident_count": self.confident_count,
            "qualified_count": self.qualified_count,
            "disputed_count": self.disputed_count,
            "abstain_count": self.abstain_count,
            "confident_rate": round(self.confident_rate, 4),
            "qualified_rate": round(self.qualified_rate, 4),
            "disputed_rate": round(self.disputed_rate, 4),
            "abstain_rate": round(self.abstain_rate, 4),
        }


@dataclass
class GovernanceFlip:
    """
    A query that changed governance mode between versions.

    Detects behavioral changes that may indicate regressions or improvements.
    Critical for evaluating system changes before deployment.

    Attributes:
        query_hash: Hash of the normalized query
        query_text: Original query text (if stored)
        old_mode: Previous governance mode
        new_mode: Current governance mode
        old_timestamp: When the previous decision was made
        new_timestamp: When the new decision was made
        old_version: Pipeline version of the old decision (for tracking)
        new_version: Pipeline version of the new decision (for tracking)
    """

    query_hash: str
    query_text: str | None
    old_mode: str
    new_mode: str
    old_timestamp: datetime
    new_timestamp: datetime
    old_version: str | None = None
    new_version: str | None = None

    @property
    def is_regression(self) -> bool:
        """
        True if the flip is likely a regression.

        Regressions:
        - CONFIDENT → ABSTAIN (was answering, now refusing)
        - CONFIDENT → DISPUTED (was clear, now conflicting)
        - QUALIFIED → ABSTAIN (was answering with caveats, now refusing)

        Improvements:
        - ABSTAIN → CONFIDENT (corpus coverage improved)
        - DISPUTED → CONFIDENT (conflict resolved)
        """
        regressions = {
            ("confident", "abstain"),
            ("confident", "disputed"),
            ("qualified", "abstain"),
        }
        return (self.old_mode, self.new_mode) in regressions

    @property
    def is_improvement(self) -> bool:
        """True if the flip is likely an improvement."""
        improvements = {
            ("abstain", "confident"),
            ("abstain", "qualified"),
            ("disputed", "confident"),
            ("disputed", "qualified"),
        }
        return (self.old_mode, self.new_mode) in improvements

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "query_hash": self.query_hash,
            "query_text": self.query_text,
            "old_mode": self.old_mode,
            "new_mode": self.new_mode,
            "old_timestamp": self.old_timestamp.isoformat(),
            "new_timestamp": self.new_timestamp.isoformat(),
            "old_version": self.old_version,
            "new_version": self.new_version,
            "is_regression": self.is_regression,
            "is_improvement": self.is_improvement,
        }


@dataclass
class ConstraintFrequency:
    """
    Frequency of a constraint being triggered.

    Helps identify which constraints are most active and may need tuning.

    Attributes:
        constraint_name: Name of the constraint
        trigger_count: Number of times triggered in the period
        total_queries: Total queries in the period (for rate calculation)
    """

    constraint_name: str
    trigger_count: int
    total_queries: int

    @property
    def trigger_rate(self) -> float:
        """Rate at which this constraint triggers."""
        if self.total_queries == 0:
            return 0.0
        return self.trigger_count / self.total_queries

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "constraint_name": self.constraint_name,
            "trigger_count": self.trigger_count,
            "trigger_rate": round(self.trigger_rate, 4),
        }


@dataclass
class AbstainTrend:
    """
    Abstain rate trend over a time bucket.

    Used for monitoring and alerting on system behavior changes.

    Attributes:
        bucket_start: Start of the time bucket
        abstain_rate: Rate of ABSTAIN decisions in the bucket
        total_queries: Number of queries in the bucket
    """

    bucket_start: datetime
    abstain_rate: float
    total_queries: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "bucket_start": self.bucket_start.isoformat(),
            "abstain_rate": round(self.abstain_rate, 4),
            "total_queries": self.total_queries,
        }


__all__ = [
    "ModeDistribution",
    "GovernanceFlip",
    "ConstraintFrequency",
    "AbstainTrend",
]
