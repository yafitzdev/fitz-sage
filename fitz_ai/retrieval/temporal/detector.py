# fitz_ai/retrieval/temporal/detector.py
"""
Temporal query detection and reference extraction.

Detects temporal intent in queries and extracts time references for:
- Comparison queries: "What changed between v1 and v2?"
- Period queries: "What was the status in Q1 2024?"
- Before/after queries: "What happened before the merger?"

Always active - no configuration needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


class TemporalIntent(Enum):
    """Type of temporal query intent."""

    NONE = "none"  # No temporal intent detected
    COMPARISON = "comparison"  # Comparing two time periods/versions
    PERIOD = "period"  # Asking about a specific time period
    BEFORE = "before"  # Asking about time before a reference
    AFTER = "after"  # Asking about time after a reference
    CHANGE = "change"  # Asking what changed (implies comparison)


@dataclass
class TemporalReference:
    """A temporal reference extracted from a query."""

    text: str  # Original text (e.g., "Q1 2024", "version 2.0", "last month")
    ref_type: str  # Type: "quarter", "year", "month", "version", "relative", "date"
    normalized: Optional[str] = None  # Normalized form if applicable


@dataclass
class TemporalDetector:
    """
    Detects temporal intent and extracts time references from queries.

    Patterns detected:
    - Quarters: Q1, Q2, Q3, Q4 (with optional year)
    - Years: 2023, 2024, etc.
    - Versions: v1, v2, version 1.0, etc.
    - Relative: last month, last year, recently, etc.
    - Comparison: between X and Y, from X to Y, X vs Y
    - Change: what changed, differences, updates
    """

    # Patterns for temporal references
    QUARTER_PATTERN: str = r"\b[Qq][1-4]\s*(?:20\d{2})?\b"
    YEAR_PATTERN: str = r"\b20[12]\d\b"
    VERSION_PATTERN: str = r"\b[Vv](?:ersion\s*)?(\d+(?:\.\d+)*)\b"
    MONTH_PATTERN: str = (
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*(?:20\d{2})?\b"
    )
    DATE_PATTERN: str = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"

    # Relative time expressions
    RELATIVE_PATTERNS: list[str] = field(
        default_factory=lambda: [
            r"\blast\s+(?:week|month|year|quarter)\b",
            r"\bprevious\s+(?:week|month|year|quarter)\b",
            r"\bthis\s+(?:week|month|year|quarter)\b",
            r"\bnext\s+(?:week|month|year|quarter)\b",
            r"\byesterday\b",
            r"\btoday\b",
            r"\brecently\b",
        ]
    )

    # Comparison indicators
    COMPARISON_PATTERNS: list[str] = field(
        default_factory=lambda: [
            r"\bbetween\s+.+\s+and\s+",
            r"\bfrom\s+.+\s+to\s+",
            r"\bvs\.?\s+",
            r"\bversus\s+",
            r"\bcompare[ds]?\s+.+\s+(?:to|with|and)\s+",
            r"\bdifference\s+(?:between|from)\s+",
        ]
    )

    # Change indicators
    CHANGE_PATTERNS: list[str] = field(
        default_factory=lambda: [
            r"\bwhat\s+(?:has\s+)?changed\b",
            r"\bwhat\s+(?:are\s+)?(?:the\s+)?changes\b",
            r"\bwhat\s+(?:is\s+)?(?:the\s+)?difference\b",
            r"\bhow\s+(?:has|did)\s+.+\s+change[ds]?\b",
            r"\bupdates?\s+(?:since|from|between)\b",
            r"\bmodifications?\s+(?:since|from|between)\b",
            r"\bevolution\s+of\b",
        ]
    )

    # Before/after indicators
    BEFORE_PATTERNS: list[str] = field(
        default_factory=lambda: [
            r"\bbefore\s+",
            r"\bprior\s+to\s+",
            r"\bearlier\s+than\s+",
            r"\bpre[-\s]",
            r"\buntil\s+",
        ]
    )

    AFTER_PATTERNS: list[str] = field(
        default_factory=lambda: [
            r"\bafter\s+",
            r"\bsince\s+",
            r"\bfollowing\s+",
            r"\bpost[-\s]",
            r"\bstarting\s+(?:from\s+)?",
        ]
    )

    def detect(self, query: str) -> tuple[TemporalIntent, list[TemporalReference]]:
        """
        Detect temporal intent and extract references from a query.

        Args:
            query: The user's query string

        Returns:
            Tuple of (intent, list of temporal references)
        """
        query_lower = query.lower()
        references = self._extract_references(query)

        # Detect intent
        intent = self._detect_intent(query_lower, references)

        if intent != TemporalIntent.NONE:
            logger.debug(
                f"Temporal detection: intent={intent.value}, "
                f"references={[r.text for r in references]}"
            )

        return intent, references

    def _detect_intent(
        self, query_lower: str, references: list[TemporalReference]
    ) -> TemporalIntent:
        """Detect the temporal intent from query patterns."""
        # Check for change/diff intent first (strongest signal)
        for pattern in self.CHANGE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return TemporalIntent.CHANGE

        # Check for comparison patterns
        for pattern in self.COMPARISON_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return TemporalIntent.COMPARISON

        # Check for before patterns
        for pattern in self.BEFORE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return TemporalIntent.BEFORE

        # Check for after patterns
        for pattern in self.AFTER_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return TemporalIntent.AFTER

        # If we have multiple temporal references, likely a comparison
        if len(references) >= 2:
            return TemporalIntent.COMPARISON

        # If we have exactly one reference, it's a period query
        if len(references) == 1:
            return TemporalIntent.PERIOD

        return TemporalIntent.NONE

    def _extract_references(self, query: str) -> list[TemporalReference]:
        """Extract all temporal references from the query."""
        references: list[TemporalReference] = []
        seen_texts: set[str] = set()

        # Extract quarters (Q1, Q2 2024, etc.)
        for match in re.finditer(self.QUARTER_PATTERN, query, re.IGNORECASE):
            text = match.group(0).strip()
            if text.lower() not in seen_texts:
                references.append(
                    TemporalReference(
                        text=text,
                        ref_type="quarter",
                        normalized=text.upper(),
                    )
                )
                seen_texts.add(text.lower())

        # Extract years
        for match in re.finditer(self.YEAR_PATTERN, query):
            text = match.group(0)
            if text not in seen_texts:
                references.append(
                    TemporalReference(
                        text=text,
                        ref_type="year",
                        normalized=text,
                    )
                )
                seen_texts.add(text)

        # Extract versions
        for match in re.finditer(self.VERSION_PATTERN, query, re.IGNORECASE):
            text = match.group(0)
            version_num = match.group(1)
            if text.lower() not in seen_texts:
                references.append(
                    TemporalReference(
                        text=text,
                        ref_type="version",
                        normalized=f"v{version_num}",
                    )
                )
                seen_texts.add(text.lower())

        # Extract months
        for match in re.finditer(self.MONTH_PATTERN, query, re.IGNORECASE):
            text = match.group(0).strip()
            if text.lower() not in seen_texts:
                references.append(
                    TemporalReference(
                        text=text,
                        ref_type="month",
                    )
                )
                seen_texts.add(text.lower())

        # Extract relative time expressions
        for pattern in self.RELATIVE_PATTERNS:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                text = match.group(0).strip()
                if text.lower() not in seen_texts:
                    references.append(
                        TemporalReference(
                            text=text,
                            ref_type="relative",
                        )
                    )
                    seen_texts.add(text.lower())

        # Extract dates
        for match in re.finditer(self.DATE_PATTERN, query):
            text = match.group(0)
            if text not in seen_texts:
                references.append(
                    TemporalReference(
                        text=text,
                        ref_type="date",
                    )
                )
                seen_texts.add(text)

        return references

    def generate_temporal_queries(
        self, query: str, intent: TemporalIntent, references: list[TemporalReference]
    ) -> list[str]:
        """
        Generate sub-queries to cover temporal aspects.

        For comparison/change queries, generates queries for each time period.
        For before/after queries, includes the temporal constraint.

        Args:
            query: Original query
            intent: Detected temporal intent
            references: Extracted temporal references

        Returns:
            List of queries to search (includes original)
        """
        queries = [query]

        if intent == TemporalIntent.NONE:
            return queries

        if intent in (TemporalIntent.COMPARISON, TemporalIntent.CHANGE):
            # For comparison, generate queries focused on each time reference
            for ref in references:
                # Create a focused query for this time period
                focused = self._create_focused_query(query, ref)
                if focused and focused not in queries:
                    queries.append(focused)

        elif intent == TemporalIntent.PERIOD:
            # For period queries, emphasize the time reference
            if references:
                ref = references[0]
                focused = self._create_focused_query(query, ref)
                if focused and focused not in queries:
                    queries.append(focused)

        elif intent in (TemporalIntent.BEFORE, TemporalIntent.AFTER):
            # For before/after, the original query is usually sufficient
            # but we can add emphasis on the reference point
            if references:
                ref = references[0]
                focused = self._create_focused_query(query, ref)
                if focused and focused not in queries:
                    queries.append(focused)

        return queries

    def _create_focused_query(self, query: str, ref: TemporalReference) -> Optional[str]:
        """Create a query focused on a specific temporal reference."""
        # For versions, create a version-specific query
        if ref.ref_type == "version":
            # Extract the core topic (remove comparison words)
            core = re.sub(
                r"\b(between|and|vs\.?|versus|compare[ds]?|difference|from|to)\b",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip()
            # Remove other version references
            core = re.sub(self.VERSION_PATTERN, "", core, flags=re.IGNORECASE).strip()
            core = re.sub(r"\s+", " ", core).strip()
            core = re.sub(r"\?+", "", core).strip()  # Remove question marks
            if core and len(core) > 3:
                return f"{core} {ref.text}"
            else:
                # If core is too short, just return the version reference
                return f"{ref.text}"

        # For quarters/years, create a time-focused query
        if ref.ref_type in ("quarter", "year", "month"):
            core = re.sub(
                r"\b(between|and|vs\.?|versus|compare[ds]?|difference|from|to)\b",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip()
            # Remove other time references
            core = re.sub(self.QUARTER_PATTERN, "", core, flags=re.IGNORECASE)
            core = re.sub(self.YEAR_PATTERN, "", core)
            core = re.sub(r"\s+", " ", core).strip()
            core = re.sub(r"\?+", "", core).strip()  # Remove question marks
            if core and len(core) > 3:
                return f"{core} {ref.text}"
            else:
                return f"{ref.text}"

        return None
