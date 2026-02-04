# fitz_ai/core/guardrails/numerical_detector.py
"""
Numerical and temporal conflict detection for dispute classification.

Distinguishes real contradictions from measurement variance.
Key insight: "10% increase" vs "12% increase" is variance, not contradiction.
But "10% increase" vs "8% decrease" IS a contradiction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class NumericMention:
    """Structured representation of a number in context."""

    value: float
    unit: Optional[str]  # %, million, GB, etc.
    direction: Optional[str]  # increase, decrease, None
    context: str  # surrounding words for entity matching
    span: tuple[int, int]  # character positions


@dataclass
class TemporalMention:
    """Structured representation of a date/time."""

    year: Optional[int]
    quarter: Optional[str]  # Q1, Q2, Q3, Q4
    context: str
    span: tuple[int, int]


class NumericalConflictDetector:
    """Detects numerical and temporal contradictions vs variance."""

    # Direction indicators
    INCREASE_PATTERNS = [
        r"\b(increased?|rises?|rising|grew|grown|growth|gains?|gained|up|higher|more|"
        r"improved?|boost|jumped?|surged?|climbed?|expanded?)\b",
    ]
    DECREASE_PATTERNS = [
        r"\b(decreased?|declines?|declining|drops?|dropped|fell|fallen|fall|down|lower|"
        r"less|reduced?|reduction|lost|losses?|shrunk|contracted?|slipped?)\b",
    ]

    # Source indicators - suggest different authoritative sources (potential dispute)
    SOURCE_PATTERNS = [
        r"\baccording to\b",
        r"\breported?\b",
        r"\bclaims?\b",
        r"\bstates?\b",
        r"\bsays?\b",
        r"\bfinds?\b",
        r"\bshows?\b",
        r"\bestimates?\b",
        r"\bsurvey\b",
        r"\bstudy\b",
        r"\bresearch\b",
        r"\banalysis\b",
        r"\binternal\b",
        r"\bexternal\b",
    ]

    # Unit patterns - capture value and unit
    UNIT_PATTERNS = [
        (r"(\d+(?:\.\d+)?)\s*(%|percent|percentage)", "%"),
        (r"(\d+(?:\.\d+)?)\s*(million|mil|M)\b", "million"),
        (r"(\d+(?:\.\d+)?)\s*(billion|bil|B)\b", "billion"),
        (r"(\d+(?:\.\d+)?)\s*(thousand|K)\b", "thousand"),
        (r"(\d+(?:\.\d+)?)\s*(GB|MB|KB|TB)", "bytes"),
        (r"\$\s*(\d+(?:\.\d+)?)\s*(million|bil|billion|M|B|K|thousand)?", "currency"),
        (r"(\d+(?:\.\d+)?)\s*(USD|EUR|dollars?|euros?)", "currency"),
    ]

    def extract_numeric_mentions(self, text: str) -> list[NumericMention]:
        """
        Extract all numeric mentions with context.

        Args:
            text: Chunk content

        Returns:
            List of structured numeric mentions
        """
        mentions = []

        for pattern, unit_type in self.UNIT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value_str = match.group(1)
                    value = float(value_str)
                except (ValueError, IndexError):
                    continue

                # Get surrounding context (50 chars before/after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                # Detect direction from context
                direction = self._detect_direction(context)

                # Normalize unit
                raw_unit = match.group(2) if match.lastindex >= 2 else None
                unit = self._normalize_unit(raw_unit, unit_type)

                mentions.append(
                    NumericMention(
                        value=value,
                        unit=unit,
                        direction=direction,
                        context=context,
                        span=(match.start(), match.end()),
                    )
                )

        return mentions

    def extract_temporal_mentions(self, text: str) -> list[TemporalMention]:
        """
        Extract temporal references.

        Args:
            text: Chunk content

        Returns:
            List of temporal mentions
        """
        mentions = []

        # Year patterns (standalone years like 2023, 2024)
        year_pattern = r"\b(19|20)\d{2}\b"
        for match in re.finditer(year_pattern, text):
            year = int(match.group())

            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]

            mentions.append(
                TemporalMention(
                    year=year,
                    quarter=None,
                    context=context,
                    span=(match.start(), match.end()),
                )
            )

        # Quarter patterns (Q1, Q2, Q3, Q4 with optional year)
        quarter_pattern = r"\b(Q[1-4])\s*(?:'?(\d{2,4}))?\b"
        for match in re.finditer(quarter_pattern, text, re.IGNORECASE):
            quarter = match.group(1).upper()
            year_str = match.group(2)
            year = None
            if year_str:
                year = int(year_str) if len(year_str) == 4 else 2000 + int(year_str)

            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]

            mentions.append(
                TemporalMention(
                    year=year,
                    quarter=quarter,
                    context=context,
                    span=(match.start(), match.end()),
                )
            )

        return mentions

    def is_numerical_variance(
        self,
        mention1: NumericMention,
        mention2: NumericMention,
        variance_threshold: float = 0.25,  # 25% relative difference
    ) -> bool:
        """
        Check if two numeric mentions are variance vs contradiction.

        Args:
            mention1: First mention
            mention2: Second mention
            variance_threshold: Max acceptable relative difference

        Returns:
            True if this is variance (not contradiction), False otherwise
        """
        # Must have same unit to compare
        if mention1.unit != mention2.unit:
            return False  # Different units = can't determine variance

        # Check directions - opposite directions = contradiction, not variance
        if mention1.direction and mention2.direction:
            if mention1.direction != mention2.direction:
                return False  # Opposite directions = contradiction

        # Calculate relative difference
        if mention1.value == 0 and mention2.value == 0:
            return True  # Both zero = same

        avg = (abs(mention1.value) + abs(mention2.value)) / 2
        if avg == 0:
            return False

        rel_diff = abs(mention1.value - mention2.value) / avg

        # Small variance = not contradiction
        return rel_diff <= variance_threshold

    def is_temporal_conflict(
        self,
        mention1: TemporalMention,
        mention2: TemporalMention,
    ) -> bool:
        """
        Check if two temporal mentions contradict.

        Args:
            mention1: First mention
            mention2: Second mention

        Returns:
            True if contradiction exists
        """
        # Both must have years to compare
        if mention1.year is None or mention2.year is None:
            return False

        # Same year = no conflict (or check quarters)
        if mention1.year == mention2.year:
            # Check quarters if both present
            if mention1.quarter and mention2.quarter:
                return mention1.quarter != mention2.quarter
            return False

        # Different years - check if describing same event via context overlap
        context_overlap = self._context_similarity(mention1.context, mention2.context)

        # High context overlap + different years = likely contradiction
        # (describing same event with different years)
        return context_overlap > 0.5

    def check_chunk_pair_variance(
        self,
        content1: str,
        content2: str,
    ) -> tuple[bool, str]:
        """
        Check if two chunks have numerical variance (not contradiction).

        This is the main entry point for conflict_aware.py integration.

        Args:
            content1: First chunk content
            content2: Second chunk content

        Returns:
            Tuple of (is_variance, reason)
            - is_variance=True means skip LLM check (it's just variance)
            - is_variance=False means proceed with LLM check
        """
        # If both chunks cite different sources, it's likely a dispute, not variance
        # e.g., "Gartner says 32%" vs "Company claims 38%" = dispute
        if self._has_conflicting_sources(content1, content2):
            return False, ""

        nums1 = self.extract_numeric_mentions(content1)
        nums2 = self.extract_numeric_mentions(content2)

        # Check all pairs for variance
        for n1 in nums1:
            for n2 in nums2:
                # Same unit required
                if n1.unit != n2.unit:
                    continue

                # Check if contexts are about the same thing
                # Use moderate threshold (0.18) to balance:
                # - Catching variance: "Sales grew 10%" vs "Sales grew 12%"
                # - Avoiding false positives: "Gartner says 32%" vs "Company claims 38%"
                if self._context_similarity(n1.context, n2.context) < 0.18:
                    continue  # Different topics

                if self.is_numerical_variance(n1, n2):
                    return (
                        True,
                        f"Numerical variance detected: {n1.value}{n1.unit or ''} vs "
                        f"{n2.value}{n2.unit or ''} (same direction, within threshold)",
                    )

        return False, ""

    def _has_conflicting_sources(self, content1: str, content2: str) -> bool:
        """
        Check if both chunks cite different sources.

        If both chunks have source indicators (e.g., "according to", "claims"),
        it suggests different authorities making claims - potential dispute.

        Returns:
            True if both chunks have source indicators (potential dispute)
        """
        content1_lower = content1.lower()
        content2_lower = content2.lower()

        has_source1 = any(
            re.search(pattern, content1_lower) for pattern in self.SOURCE_PATTERNS
        )
        has_source2 = any(
            re.search(pattern, content2_lower) for pattern in self.SOURCE_PATTERNS
        )

        # Both chunks citing sources = likely a dispute between sources
        return has_source1 and has_source2

    def _detect_direction(self, context: str) -> Optional[str]:
        """Detect direction from context."""
        context_lower = context.lower()

        for pattern in self.INCREASE_PATTERNS:
            if re.search(pattern, context_lower):
                return "increase"

        for pattern in self.DECREASE_PATTERNS:
            if re.search(pattern, context_lower):
                return "decrease"

        return None

    def _normalize_unit(self, raw_unit: Optional[str], unit_type: str) -> str:
        """Normalize unit to canonical form."""
        if unit_type == "currency":
            return "currency"
        if unit_type == "bytes":
            return "bytes"
        if raw_unit:
            raw_lower = raw_unit.lower()
            if raw_lower in ("%", "percent", "percentage"):
                return "%"
            if raw_lower in ("million", "mil", "m"):
                return "million"
            if raw_lower in ("billion", "bil", "b"):
                return "billion"
            if raw_lower in ("thousand", "k"):
                return "thousand"
        return unit_type

    def _context_similarity(self, context1: str, context2: str) -> float:
        """
        Simple token overlap for context similarity.

        Returns value 0-1.
        """
        # Remove numbers and common words for better topic matching
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "was",
            "were",
            "are",
            "be",
            "been",
            "by",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "from",
            "and",
            "or",
            "that",
            "this",
            "it",
        }

        def tokenize(text: str) -> set[str]:
            # Remove numbers, keep only alpha tokens
            tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
            return {t for t in tokens if t not in stopwords}

        tokens1 = tokenize(context1)
        tokens2 = tokenize(context2)

        if not tokens1 or not tokens2:
            return 0.0

        overlap = len(tokens1 & tokens2)
        total = len(tokens1 | tokens2)

        return overlap / total if total > 0 else 0.0


__all__ = ["NumericalConflictDetector", "NumericMention", "TemporalMention"]
