# fitz_ai/core/conflicts.py
"""
Conflict Detection - Core epistemic capability for fitz-ai.

Detects contradictory claims across text chunks. This is a platform-wide
capability that supports epistemic honesty - a core trait of fitz-ai.

Used by:
- Query-time constraints (engines/classic_rag/constraints/)
- Hierarchy enrichment (ingest/enrichment/hierarchy/)
- Any future engine that needs conflict awareness

Detects conflicts in:
- Classifications (security vs operational incident)
- Trends (improved vs declined)
- Sentiment (positive vs negative)
- State (successful vs failed, approved vs rejected)
- Numeric claims (significantly different values)
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class ChunkLike(Protocol):
    """Protocol for objects that have id and content."""

    @property
    def id(self) -> str: ...

    @property
    def content(self) -> str: ...


# =============================================================================
# Conflict Detection Patterns
# =============================================================================

# Patterns for extracting claims from text
CLAIM_PATTERNS: tuple[re.Pattern, ...] = (
    # Incident/event classification
    re.compile(
        r"\b(?:is|was|are|were)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)?)\s+(?:incident|event|issue|problem)",
        re.I,
    ),
    re.compile(r"\bclassified\s+as\s+(?:a|an)?\s*['\"]?(\w+(?:\s+\w+)?)['\"]?", re.I),
    re.compile(r"\bcategor(?:y|ized)\s+(?:as|:)\s*['\"]?(\w+(?:\s+\w+)?)['\"]?", re.I),
    re.compile(r"\btype\s*:\s*['\"]?(\w+(?:\s+\w+)?)['\"]?", re.I),
    re.compile(
        r"\b(?:security|operational|infrastructure|network)\s+(?:incident|event|issue)",
        re.I,
    ),
    # Trend indicators
    re.compile(r"\b(improved|increased|grew|rose|gained|surged)\b", re.I),
    re.compile(r"\b(declined|decreased|dropped|fell|lost|plummeted)\b", re.I),
    re.compile(r"\b(stable|unchanged|flat|steady|consistent)\b", re.I),
    # Sentiment indicators
    re.compile(r"\b(?:was|is|were|are)\s+(positive|negative|neutral)\b", re.I),
    re.compile(r"\b(?:response|feedback|sentiment)\s+(?:was|is)\s+(positive|negative|neutral)\b", re.I),
    re.compile(r"\b(good|bad|excellent|poor|great|terrible)\s+(?:results?|performance|outcome)\b", re.I),
    # State indicators (both "was successful" and standalone "failed")
    re.compile(r"\b(?:was|is|were|are)\s+(successful|failed|completed|pending)\b", re.I),
    re.compile(r"\b(successful(?:ly)?|failed|completed|pending)\b", re.I),
    re.compile(r"\b(?:was|is|were|are)\s+(approved|rejected|accepted|denied)\b", re.I),
    # Numeric claims (NPS, scores, percentages)
    re.compile(r"\b(?:nps|score|rating)\s*(?:is|was|of|:)?\s*(\d+(?:\.\d+)?)\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*(?:increase|decrease|growth|decline)?\b", re.I),
)

# Mutually exclusive pairs (normalized to lowercase)
CONFLICTING_PAIRS: tuple[tuple[str, str], ...] = (
    # Classification conflicts
    ("security", "operational"),
    ("security incident", "operational incident"),
    ("internal", "external"),
    # State conflicts
    ("confirmed", "unconfirmed"),
    ("approved", "rejected"),
    ("accepted", "denied"),
    ("active", "deprecated"),
    ("enabled", "disabled"),
    ("primary", "secondary"),
    ("successful", "failed"),
    ("completed", "pending"),
    ("yes", "no"),
    ("true", "false"),
    # Trend conflicts
    ("improved", "declined"),
    ("improved", "decreased"),
    ("improved", "dropped"),
    ("improved", "fell"),
    ("increased", "decreased"),
    ("increased", "declined"),
    ("increased", "dropped"),
    ("increased", "fell"),
    ("grew", "dropped"),
    ("grew", "declined"),
    ("grew", "decreased"),
    ("rose", "fell"),
    ("rose", "dropped"),
    ("rose", "declined"),
    ("gained", "lost"),
    ("surged", "plummeted"),
    ("stable", "improved"),
    ("stable", "declined"),
    ("stable", "increased"),
    ("stable", "decreased"),
    ("unchanged", "improved"),
    ("unchanged", "declined"),
    # Sentiment conflicts
    ("positive", "negative"),
    ("good", "bad"),
    ("good", "poor"),
    ("excellent", "poor"),
    ("excellent", "terrible"),
    ("great", "terrible"),
)


# =============================================================================
# Core Functions
# =============================================================================


def extract_claims(text: str) -> set[str]:
    """
    Extract claims from text that could potentially conflict.

    Returns a set of normalized claim terms found in the text.
    """
    claims: set[str] = set()

    for pattern in CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            term = match.group(1) if match.lastindex else match.group(0)
            if term:
                claims.add(term.lower().strip())

    return claims


def are_conflicting(term1: str, term2: str) -> bool:
    """
    Check if two claim terms are mutually exclusive.

    Returns True if the terms represent conflicting claims.
    """
    t1, t2 = term1.lower(), term2.lower()

    # Same term is not a conflict
    if t1 == t2:
        return False

    # Check explicit conflicting pairs
    for a, b in CONFLICTING_PAIRS:
        if (a in t1 and b in t2) or (b in t1 and a in t2):
            return True

    # Check numeric conflicts (significantly different values)
    num1 = re.search(r"(\d+(?:\.\d+)?)", t1)
    num2 = re.search(r"(\d+(?:\.\d+)?)", t2)
    if num1 and num2:
        try:
            val1, val2 = float(num1.group(1)), float(num2.group(1))
            # Significant difference (>20% relative difference)
            if val1 > 0 and val2 > 0:
                ratio = max(val1, val2) / min(val1, val2)
                if ratio > 1.2:
                    return True
        except ValueError:
            pass

    return False


def find_conflicts(chunks: Sequence[ChunkLike]) -> list[tuple[str, str, str, str]]:
    """
    Find conflicting claims across chunks.

    Args:
        chunks: Sequence of objects with 'id' and 'content' attributes

    Returns:
        List of (chunk1_id, claim1, chunk2_id, claim2) tuples representing
        each detected conflict.
    """
    conflicts: list[tuple[str, str, str, str]] = []

    # Extract claims per chunk
    chunk_claims: list[tuple[str, set[str]]] = []
    for chunk in chunks:
        claims = extract_claims(chunk.content)
        if claims:
            chunk_claims.append((chunk.id, claims))

    # Compare all pairs
    for i, (id1, claims1) in enumerate(chunk_claims):
        for id2, claims2 in chunk_claims[i + 1 :]:
            for term1 in claims1:
                for term2 in claims2:
                    if are_conflicting(term1, term2):
                        conflicts.append((id1, term1, id2, term2))

    return conflicts


__all__ = [
    "ChunkLike",
    "CLAIM_PATTERNS",
    "CONFLICTING_PAIRS",
    "extract_claims",
    "are_conflicting",
    "find_conflicts",
]
