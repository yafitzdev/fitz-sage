# fitz_ai/evaluation/__init__.py
"""
Governance Observability - Track and analyze governance decisions.

This module provides tools for:
- Logging governance decisions to PostgreSQL
- Aggregating mode distributions over time
- Detecting behavioral changes (flips) between versions
- CLI commands for governance statistics

Usage:
    from fitz_ai.evaluation import GovernanceLogger, GovernanceStats

    # Log a decision
    logger = GovernanceLogger(pool, collection="default")
    log_entry = logger.log(decision, query, chunks)

    # Query statistics
    stats = GovernanceStats(pool)
    distribution = stats.get_mode_distribution(days=7)
    print(f"Abstain rate: {distribution.abstain_rate:.1%}")
"""

from .models import (
    AbstainTrend,
    ConstraintFrequency,
    GovernanceFlip,
    ModeDistribution,
)

__all__ = [
    # Models
    "ModeDistribution",
    "GovernanceFlip",
    "ConstraintFrequency",
    "AbstainTrend",
]


# Lazy imports for optional components (require DB connection)
def __getattr__(name: str):
    if name == "GovernanceLogger":
        from .logger import GovernanceLogger

        return GovernanceLogger
    if name == "GovernanceStats":
        from .stats import GovernanceStats

        return GovernanceStats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
