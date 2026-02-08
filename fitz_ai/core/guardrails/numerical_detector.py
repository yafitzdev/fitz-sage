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

    # Number pattern fragment: matches "42.8", "165,000", "299,792,458"
    _NUM = r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"

    # Unit patterns - capture value and unit
    # Order matters: compound units before single-letter ambiguities (m/s before M=million)
    UNIT_PATTERNS = [
        (rf"{_NUM}\s*(%|percent|percentage)", "%"),
        (rf"{_NUM}\s*(GB|MB|KB|TB)", "bytes"),
        # Compound units before single-letter multipliers
        (rf"{_NUM}\s*(m/s|km/h|mph|ft/s)", "measurement"),
        (rf"{_NUM}\s*(million|mil)\b", "million"),
        (rf"{_NUM}\s*(billion|bil)\b", "billion"),
        (rf"{_NUM}\s*(thousand)\b", "thousand"),
        # Single-letter multipliers after compound units
        (rf"{_NUM}\s+(M)\b", "million"),
        (rf"{_NUM}\s+(B)\b", "billion"),
        (rf"{_NUM}\s+(K)\b", "thousand"),
        (rf"\$\s*{_NUM}\s*(million|bil|billion|M|B|K|thousand)?", "currency"),
        (rf"{_NUM}\s*(USD|EUR|dollars?|euros?)", "currency"),
        # Physical units (e.g., "299,792,458 meters", "1.1°C")
        (rf"{_NUM}\s*(meters?|km|miles?|kg|lbs?|°[CF]|degrees?)", "measurement"),
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
                    value_str = match.group(1).replace(",", "")
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
        variance_threshold: float = 0.05,  # 5% relative difference
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

        Multi-source convergence: when different sources cite nearly identical
        numbers (e.g., Gartner $42.8B vs IDC $43.1B), that's agreement, not
        dispute. The source check only blocks variance when numbers differ
        significantly.

        Args:
            content1: First chunk content
            content2: Second chunk content

        Returns:
            Tuple of (is_variance, reason)
            - is_variance=True means skip LLM check (it's just variance)
            - is_variance=False means proceed with LLM check
        """
        nums1 = self.extract_numeric_mentions(content1)
        nums2 = self.extract_numeric_mentions(content2)

        # Count mentions per unit in each chunk. When a chunk contains
        # multiple values of the same unit (e.g., "85%, 95%, 80%, 90%"),
        # we can't reliably determine which value is "the" stat being
        # reported. Be conservative: only detect variance when each chunk
        # has exactly one mention per unit.
        from collections import Counter

        units1 = Counter(n.unit for n in nums1 if n.unit)
        units2 = Counter(n.unit for n in nums2 if n.unit)
        ambiguous_units = {u for u, c in units1.items() if c > 1} | {
            u for u, c in units2.items() if c > 1
        }

        for n1 in nums1:
            for n2 in nums2:
                if n1.unit != n2.unit:
                    continue

                if n1.unit in ambiguous_units:
                    continue

                if not self.is_numerical_variance(n1, n2):
                    continue

                return (
                    True,
                    f"Numerical variance detected: {n1.value}{n1.unit or ''} vs "
                    f"{n2.value}{n2.unit or ''} (same direction, within threshold)",
                )

        return False, ""

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
        if unit_type == "measurement":
            if raw_unit:
                raw_lower = raw_unit.lower().rstrip("s")
                if raw_lower in ("meter", "m", "m/"):
                    return "meters"
                if raw_lower in ("m/s", "ft/s"):
                    return "speed"
                if raw_lower in ("km/h", "mph"):
                    return "speed"
                if raw_lower in ("km", "kilometer"):
                    return "km"
                if raw_lower in ("mile"):
                    return "miles"
                if raw_lower in ("kg", "kilogram"):
                    return "kg"
                if raw_lower in ("lb"):
                    return "lbs"
                if raw_lower in ("°c", "°f", "degree"):
                    return raw_lower
            return "measurement"
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
