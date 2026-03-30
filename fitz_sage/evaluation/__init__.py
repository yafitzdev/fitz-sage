# fitz_sage/evaluation/__init__.py
"""
Governance Observability and Benchmarking.

This module provides tools for:
- Logging governance decisions to PostgreSQL
- Aggregating mode distributions over time
- Detecting behavioral changes (flips) between versions
- CLI commands for governance statistics
- Benchmarking against industry standards (BEIR)
- Fitz-native governance benchmarks (fitz-gov)

Usage:
    from fitz_sage.evaluation import GovernanceLogger, GovernanceStats

    # Log a decision
    logger = GovernanceLogger(pool, collection="default")
    log_entry = logger.log(decision, query, chunks)

    # Query statistics
    stats = GovernanceStats(pool)
    distribution = stats.get_mode_distribution(days=7)
    print(f"Abstain rate: {distribution.abstain_rate:.1%}")

Benchmarks:
    from fitz_sage.evaluation.benchmarks import BEIRBenchmark, FitzGovBenchmark

    # Run BEIR retrieval benchmark (cross-RAG comparison)
    beir = BEIRBenchmark()
    results = beir.evaluate(engine, dataset="scifact")

    # Run fitz-gov governance benchmark (Fitz's moat)
    gov = FitzGovBenchmark()
    results = gov.evaluate(engine)
"""

from .models import (
    AbstainTrend,
    ConstraintFrequency,
    GovernanceFlip,
    ModeDistribution,
)

__all__ = [
    # Governance Models
    "ModeDistribution",
    "GovernanceFlip",
    "ConstraintFrequency",
    "AbstainTrend",
    # Governance (lazy)
    "GovernanceLogger",
    "GovernanceStats",
    # Benchmarks (lazy)
    "BEIRResult",
    "BEIRBenchmark",
    "RGBTestType",
    "RGBCase",
    "RGBResult",
    "RGBEvaluator",
    "FitzGovCategory",
    "FitzGovCase",
    "FitzGovCaseResult",
    "FitzGovCategoryResult",
    "FitzGovResult",
    "FitzGovBenchmark",
    # Dashboard
    "BenchmarkDashboard",
]


# Lazy imports for optional components
def __getattr__(name: str):
    # Governance components (require DB connection)
    if name == "GovernanceLogger":
        from .logger import GovernanceLogger

        return GovernanceLogger
    if name == "GovernanceStats":
        from .stats import GovernanceStats

        return GovernanceStats

    # Dashboard
    if name == "BenchmarkDashboard":
        from .dashboard import BenchmarkDashboard

        return BenchmarkDashboard

    # BEIR
    if name == "BEIRResult":
        from .benchmarks.beir import BEIRResult

        return BEIRResult
    if name == "BEIRBenchmark":
        from .benchmarks.beir import BEIRBenchmark

        return BEIRBenchmark

    # RGB
    if name == "RGBTestType":
        from .benchmarks.rgb import RGBTestType

        return RGBTestType
    if name == "RGBCase":
        from .benchmarks.rgb import RGBCase

        return RGBCase
    if name == "RGBResult":
        from .benchmarks.rgb import RGBResult

        return RGBResult
    if name == "RGBEvaluator":
        from .benchmarks.rgb import RGBEvaluator

        return RGBEvaluator

    # fitz-gov
    if name == "FitzGovCategory":
        from .benchmarks.fitz_gov import FitzGovCategory

        return FitzGovCategory
    if name == "FitzGovCase":
        from .benchmarks.fitz_gov import FitzGovCase

        return FitzGovCase
    if name == "FitzGovCaseResult":
        from .benchmarks.fitz_gov import FitzGovCaseResult

        return FitzGovCaseResult
    if name == "FitzGovCategoryResult":
        from .benchmarks.fitz_gov import FitzGovCategoryResult

        return FitzGovCategoryResult
    if name == "FitzGovResult":
        from .benchmarks.fitz_gov import FitzGovResult

        return FitzGovResult
    if name == "FitzGovBenchmark":
        from .benchmarks.fitz_gov import FitzGovBenchmark

        return FitzGovBenchmark

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
