# fitz_ai/retrieval/aggregation/detector.py
"""
Aggregation query detector.

Detects when a query is asking for aggregated information:
- LIST: Enumerate items ("list all X", "what are the Y")
- COUNT: Count items ("how many X", "number of Y")
- UNIQUE: Find distinct values ("different types of", "unique X")

When detected, retrieval expands to gather comprehensive coverage,
and the query is augmented to encourage exhaustive listing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


class AggregationType(Enum):
    """Type of aggregation requested."""

    LIST = auto()  # "list all", "enumerate", "what are the"
    COUNT = auto()  # "how many", "count", "number of"
    UNIQUE = auto()  # "different", "distinct", "types of"


@dataclass
class AggregationIntent:
    """Detected aggregation intent from a query."""

    type: AggregationType
    target: str  # What's being aggregated (e.g., "test cases", "errors")
    confidence: float  # 0-1 confidence in detection


@dataclass
class AggregationResult:
    """Result of aggregation detection."""

    detected: bool
    intent: AggregationIntent | None = None
    augmented_query: str | None = None  # Query rewritten for better retrieval
    fetch_multiplier: int = 1  # How many more chunks to fetch (e.g., 3x normal)


class AggregationDetector:
    """
    Detects aggregation intent in queries.

    Aggregation queries need special handling:
    1. Fetch more chunks for comprehensive coverage
    2. Augment query to encourage exhaustive results
    3. Signal to RGS that a list/count format is expected

    Usage:
        detector = AggregationDetector()
        result = detector.detect("list all test cases that failed")

        if result.detected:
            # Fetch more chunks
            top_k = base_top_k * result.fetch_multiplier
            # Use augmented query for retrieval
            query = result.augmented_query
    """

    # Patterns for LIST intent
    LIST_PATTERNS = [
        (r"\b(list|enumerate|show)\s+(all|every|each)\b", 0.95),
        (r"\bwhat\s+are\s+(all\s+)?(the\s+)?(different\s+)?", 0.85),
        (r"\bgive\s+me\s+(a\s+)?list\b", 0.90),
        (r"\b(all|every)\s+\w+\s+(that|which|where)\b", 0.80),
        (r"\bwhat\s+\w+\s+are\s+there\b", 0.85),
        (r"\bwhich\s+(all\s+)?\w+\b", 0.70),
    ]

    # Patterns for COUNT intent
    COUNT_PATTERNS = [
        (r"\bhow\s+many\b", 0.95),
        (r"\b(count|total|number)\s+(of|the)\b", 0.90),
        (r"\bhow\s+much\b", 0.80),
        (r"\bquantity\s+of\b", 0.85),
    ]

    # Patterns for UNIQUE intent
    UNIQUE_PATTERNS = [
        (r"\b(different|distinct|unique)\s+(types?|kinds?|categories)\b", 0.95),
        (r"\bwhat\s+(types?|kinds?|categories)\s+of\b", 0.90),
        (r"\b(types?|kinds?|varieties)\s+of\b", 0.75),
        (r"\bdistinct\s+\w+\b", 0.85),
    ]

    def __init__(self) -> None:
        """Initialize the detector with compiled patterns."""
        self._list_patterns = [(re.compile(p, re.I), c) for p, c in self.LIST_PATTERNS]
        self._count_patterns = [(re.compile(p, re.I), c) for p, c in self.COUNT_PATTERNS]
        self._unique_patterns = [(re.compile(p, re.I), c) for p, c in self.UNIQUE_PATTERNS]

    def detect(self, query: str) -> AggregationResult:
        """
        Detect aggregation intent in a query.

        Args:
            query: User query string

        Returns:
            AggregationResult with detection info
        """
        # Check each type in order of specificity
        for agg_type, patterns in [
            (AggregationType.COUNT, self._count_patterns),
            (AggregationType.UNIQUE, self._unique_patterns),
            (AggregationType.LIST, self._list_patterns),
        ]:
            for pattern, confidence in patterns:
                match = pattern.search(query)
                if match:
                    target = self._extract_target(query, match)
                    intent = AggregationIntent(
                        type=agg_type,
                        target=target,
                        confidence=confidence,
                    )

                    augmented = self._augment_query(query, intent)
                    multiplier = self._get_fetch_multiplier(agg_type)

                    logger.debug(
                        f"[AGGREGATION] Detected {agg_type.name} intent "
                        f"(target='{target}', confidence={confidence:.2f})"
                    )

                    return AggregationResult(
                        detected=True,
                        intent=intent,
                        augmented_query=augmented,
                        fetch_multiplier=multiplier,
                    )

        return AggregationResult(detected=False)

    def _extract_target(self, query: str, match: re.Match) -> str:
        """Extract what's being aggregated from the query."""
        # Get text after the match
        after_match = query[match.end() :].strip()

        # Extract noun phrase (simple heuristic: take words until punctuation or stop word)
        stop_words = {"that", "which", "where", "when", "in", "on", "for", "with", "from", "?"}
        words = []
        for word in after_match.split():
            clean = word.strip("?.,!").lower()
            if clean in stop_words:
                break
            words.append(word.strip("?.,!"))
            if len(words) >= 4:  # Limit target length
                break

        return " ".join(words) if words else "items"

    def _augment_query(self, query: str, intent: AggregationIntent) -> str:
        """
        Augment the query for better retrieval coverage.

        Adds explicit instructions to encourage comprehensive results.
        """
        base = query.rstrip("?").strip()

        if intent.type == AggregationType.COUNT:
            return f"{base} (include all instances for accurate count)"
        elif intent.type == AggregationType.UNIQUE:
            return f"{base} (include all different types and categories)"
        else:  # LIST
            return f"{base} (include complete list of all {intent.target})"

    def _get_fetch_multiplier(self, agg_type: AggregationType) -> int:
        """Get how many times more chunks to fetch for this aggregation type."""
        # COUNT needs most coverage for accuracy
        # LIST needs comprehensive results
        # UNIQUE needs diverse examples
        return {
            AggregationType.COUNT: 4,
            AggregationType.LIST: 3,
            AggregationType.UNIQUE: 3,
        }.get(agg_type, 2)

    def get_retrieval_boost(self, intent: AggregationIntent) -> dict:
        """
        Get retrieval parameters to optimize for this aggregation.

        Returns dict with:
        - fetch_multiplier: How many more chunks to fetch
        - diversity_boost: Whether to boost source diversity
        - dedup_threshold: Similarity threshold for deduplication
        """
        if intent.type == AggregationType.COUNT:
            return {
                "fetch_multiplier": 4,
                "diversity_boost": True,
                "dedup_threshold": 0.95,  # Strict dedup for accurate count
            }
        elif intent.type == AggregationType.UNIQUE:
            return {
                "fetch_multiplier": 3,
                "diversity_boost": True,
                "dedup_threshold": 0.80,  # Allow more variety
            }
        else:  # LIST
            return {
                "fetch_multiplier": 3,
                "diversity_boost": True,
                "dedup_threshold": 0.90,
            }
