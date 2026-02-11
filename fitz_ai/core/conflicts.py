# fitz_ai/core/conflicts.py
"""
Conflict Detection - Core epistemic capability for fitz-ai.

Conflict detection is performed at QUERY TIME using LLM-based analysis
in ConflictAwareConstraint. This module provides an empty stub for
ingest-time compatibility.

Architecture:
- Query-time: ConflictAwareConstraint uses fast LLM to detect contradictions
- Ingest-time: No conflict detection (too expensive, deferred to query time)

The LLM-based approach was chosen because embedding-based conflict detection
(cosine similarity) cannot reliably distinguish between:
- Complementary information (should NOT be flagged)
- Contradictory claims (SHOULD be flagged)

See fitz_ai.engines.fitz_rag.guardrails.plugins.conflict_aware for the LLM implementation.
"""

from __future__ import annotations

from typing import Sequence

from fitz_ai.core.chunk import Chunk


def find_conflicts(chunks: Sequence[Chunk]) -> list[tuple[str, str, str, str]]:
    """
    Ingest-time conflict detection stub.

    Returns an empty list. Actual conflict detection happens at query time
    using LLM-based analysis in ConflictAwareConstraint.

    This stub exists for hierarchy enricher compatibility during ingestion.

    Args:
        chunks: Sequence of chunks to analyze (unused)

    Returns:
        Empty list (conflict detection deferred to query time)
    """
    return []


__all__ = ["find_conflicts"]
