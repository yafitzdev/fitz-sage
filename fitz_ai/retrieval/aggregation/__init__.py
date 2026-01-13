# fitz_ai/retrieval/aggregation/__init__.py
"""
Aggregation query detection and handling.

Detects and handles queries asking for lists, counts, or enumerations:
- LIST: "list all test cases", "what are the different errors"
- COUNT: "how many users", "count the failures"
- UNIQUE: "what types of", "distinct categories"
"""

from fitz_ai.retrieval.aggregation.detector import (
    AggregationDetector,
    AggregationIntent,
    AggregationResult,
    AggregationType,
)

__all__ = [
    "AggregationDetector",
    "AggregationIntent",
    "AggregationResult",
    "AggregationType",
]
