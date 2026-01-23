# fitz_ai/retrieval/detection/__init__.py
"""
Unified detection system for query pattern matching.

Uses LLM classification for robust detection:
- Temporal queries (time periods, versions, dates)
- Aggregation queries (list, count, unique)
- Comparison queries (vs, compare, difference)
- Freshness signals (recency, authority keywords)
- Rewriter signals (pronouns, compound queries)

Dict-based expansion for query variations:
- Synonym substitution
- Acronym expansion

Usage:
    from fitz_ai.retrieval.detection import DetectionOrchestrator

    orchestrator = DetectionOrchestrator(chat_client=chat)
    summary = orchestrator.detect_for_retrieval(query)

    if summary.has_temporal_intent:
        # Route to temporal strategy
        ...
"""

from .detectors import ExpansionDetector
from .llm_classifier import LLMClassifier
from .protocol import DetectionCategory, DetectionResult, Match
from .registry import (
    AggregationType,
    DetectionOrchestrator,
    DetectionSummary,
    TemporalIntent,
)

__all__ = [
    # Orchestration
    "DetectionCategory",
    "DetectionOrchestrator",
    "DetectionResult",
    "DetectionSummary",
    # Enums
    "AggregationType",
    "TemporalIntent",
    # Protocol
    "Match",
    # LLM Classifier
    "LLMClassifier",
    # Detectors
    "ExpansionDetector",
]
