# fitz_sage/evaluation/benchmarks/__init__.py
"""
Benchmark suite for evaluating Fitz RAG quality.

Industry-standard benchmark:
- BEIR: Retrieval quality (nDCG@10, Recall@100) for cross-RAG comparison

Fitz-native benchmarks:
- RGB: Robustness (noise, rejection, conflicts)
- fitz-gov: Comprehensive governance calibration (Fitz's differentiator)
  - abstention, dispute, trustworthy_hedged, trustworthy_direct (governance modes)
  - grounding, relevance (answer quality)

Usage:
    from fitz_sage.evaluation.benchmarks import BEIRBenchmark, FitzGovBenchmark

    # Run BEIR on a dataset (cross-RAG comparison)
    beir = BEIRBenchmark()
    results = beir.evaluate(engine, dataset="scifact")

    # Run fitz-gov governance tests (Fitz's moat)
    gov = FitzGovBenchmark()
    results = gov.evaluate(engine)
"""

__all__ = [
    # BEIR
    "BEIRResult",
    "BEIRSuiteResult",
    "BEIRBenchmark",
    # RGB
    "RGBTestType",
    "RGBCase",
    "RGBResult",
    "RGBEvaluator",
    # fitz-gov
    "FitzGovCategory",
    "FitzGovCase",
    "FitzGovCaseResult",
    "FitzGovCategoryResult",
    "FitzGovResult",
    "FitzGovBenchmark",
    # LLM Validator
    "OllamaValidator",
    "ValidatorConfig",
    "ValidationResult",
]


def __getattr__(name: str):
    """Lazy import benchmark classes."""
    # BEIR
    if name in ("BEIRResult", "BEIRSuiteResult", "BEIRBenchmark"):
        from .beir import BEIRBenchmark, BEIRResult, BEIRSuiteResult

        if name == "BEIRResult":
            return BEIRResult
        if name == "BEIRSuiteResult":
            return BEIRSuiteResult
        return BEIRBenchmark

    # RGB
    if name in ("RGBTestType", "RGBCase", "RGBResult", "RGBEvaluator"):
        from .rgb import RGBCase, RGBEvaluator, RGBResult, RGBTestType

        if name == "RGBTestType":
            return RGBTestType
        elif name == "RGBCase":
            return RGBCase
        elif name == "RGBResult":
            return RGBResult
        return RGBEvaluator

    # fitz-gov - FitzGovBenchmark is the main entry point
    if name == "FitzGovBenchmark":
        from .fitz_gov import FitzGovBenchmark

        return FitzGovBenchmark

    # fitz-gov types and LLM Validator - import from fitz-gov package
    if name in (
        "FitzGovCategory",
        "FitzGovCase",
        "FitzGovCaseResult",
        "FitzGovCategoryResult",
        "FitzGovResult",
        "OllamaValidator",
        "ValidatorConfig",
        "ValidationResult",
    ):
        try:
            import fitz_gov

            return getattr(fitz_gov, name)
        except ImportError:
            raise ImportError(
                f"fitz-gov package is required for {name}. "
                "Install with: pip install fitz-gov "
                "or: pip install -e path/to/fitz-gov"
            )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
