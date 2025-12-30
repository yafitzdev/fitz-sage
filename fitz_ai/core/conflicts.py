# fitz_ai/core/conflicts.py
"""
Conflict Detection - Core epistemic capability for fitz-ai.

This module provides backward compatibility for code using the old
regex-based conflict detection API.

The actual conflict detection logic is now in:
- fitz_ai.core.guardrails.semantic.SemanticMatcher

Used by:
- Query-time constraints (guardrails/)
- Ingest-time enrichment (hierarchy/)
- Any engine that needs conflict awareness

For new code, prefer using SemanticMatcher directly.
"""

from __future__ import annotations

from typing import Sequence

# Re-export ChunkLike from its canonical location
from fitz_ai.core.chunk import ChunkLike


def find_conflicts(chunks: Sequence[ChunkLike]) -> list[tuple[str, str, str, str]]:
    """
    Backward-compatible conflict detection stub.

    This function returns an empty list for compatibility.
    For actual conflict detection, use SemanticMatcher.find_conflicts()
    which provides language-agnostic semantic conflict detection.

    The ingestion pipeline can optionally be configured to use
    semantic conflict detection by providing an embedder.

    Args:
        chunks: Sequence of chunks to analyze

    Returns:
        Empty list (no conflicts detected without semantic matcher)

    Note:
        This is a compatibility stub. For production use, integrate
        SemanticMatcher with an embedder for semantic conflict detection.
    """
    # Stub: return empty conflicts for compatibility
    # Semantic conflict detection requires an embedder which is not
    # available in this legacy API. Use SemanticMatcher instead.
    return []


__all__ = ["ChunkLike", "find_conflicts"]
