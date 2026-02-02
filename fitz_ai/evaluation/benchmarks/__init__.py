# fitz_ai/evaluation/benchmarks/__init__.py
"""
Benchmark suite for evaluating Fitz RAG quality.

Industry-standard benchmark:
- BEIR: Retrieval quality (nDCG@10, Recall@100) for cross-RAG comparison

Fitz-native benchmarks:
- RGB: Robustness (noise, rejection, conflicts)
- FITZ-GOV: Comprehensive governance calibration (Fitz's differentiator)
  - abstention, dispute, qualification, confidence (governance modes)
  - grounding, relevance (answer quality)

Usage:
    from fitz_ai.evaluation.benchmarks import BEIRBenchmark, FitzGovBenchmark

    # Run BEIR on a dataset (cross-RAG comparison)
    beir = BEIRBenchmark()
    results = beir.evaluate(engine, dataset="scifact")

    # Run FITZ-GOV governance tests (Fitz's moat)
    gov = FitzGovBenchmark()
    results = gov.evaluate(engine)
"""

__all__ = [
    # BEIR
    "BEIRResult",
    "BEIRBenchmark",
    # RGB
    "RGBTestType",
    "RGBCase",
    "RGBResult",
    "RGBEvaluator",
    # FITZ-GOV
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
    if name in ("BEIRResult", "BEIRBenchmark"):
        from .beir import BEIRBenchmark, BEIRResult

        return BEIRResult if name == "BEIRResult" else BEIRBenchmark

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

    # FITZ-GOV
    if name in (
        "FitzGovCategory",
        "FitzGovCase",
        "FitzGovCaseResult",
        "FitzGovCategoryResult",
        "FitzGovResult",
        "FitzGovBenchmark",
    ):
        from .fitz_gov import (
            FitzGovBenchmark,
            FitzGovCase,
            FitzGovCaseResult,
            FitzGovCategory,
            FitzGovCategoryResult,
            FitzGovResult,
        )

        if name == "FitzGovCategory":
            return FitzGovCategory
        elif name == "FitzGovCase":
            return FitzGovCase
        elif name == "FitzGovCaseResult":
            return FitzGovCaseResult
        elif name == "FitzGovCategoryResult":
            return FitzGovCategoryResult
        elif name == "FitzGovResult":
            return FitzGovResult
        return FitzGovBenchmark

    # LLM Validator
    if name in ("OllamaValidator", "ValidatorConfig", "ValidationResult"):
        from .llm_validator import OllamaValidator, ValidationResult, ValidatorConfig

        if name == "OllamaValidator":
            return OllamaValidator
        elif name == "ValidatorConfig":
            return ValidatorConfig
        return ValidationResult

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
