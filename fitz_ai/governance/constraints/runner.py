# fitz_ai/governance/constraints/runner.py
"""
Constraint Runner - Applies constraint plugins to retrieved context.

Delegates to the staged pipeline for hierarchical execution:
  Stage 1 (Relevance) → Stage 2 (Sufficiency) → Stage 3 (Consistency)

Returns individual results from each constraint (not combined).
Signal aggregation happens in the governance layer, not here.
"""

from __future__ import annotations

from typing import Sequence

from fitz_ai.governance.protocol import EvidenceItem

from .base import ConstraintPlugin, ConstraintResult
from .staged import run_staged_constraints


def run_constraints(
    query: str,
    chunks: Sequence[EvidenceItem],
    constraints: Sequence[ConstraintPlugin],
) -> list[ConstraintResult]:
    """
    Apply all constraints and return individual results.

    Uses staged execution: relevance → sufficiency → consistency.
    Short-circuits on abstain (skips conflict detection on irrelevant content).

    Each constraint's result is preserved separately so signals
    are not lost during aggregation. Signal resolution happens
    in the governance layer via AnswerGovernor.decide().

    Args:
        query: The user's question
        chunks: Retrieved evidence items (post-retrieval, pre-generation)
        constraints: List of constraint plugins to apply

    Returns:
        List of individual ConstraintResult objects (one per constraint)
    """
    return run_staged_constraints(query, chunks, constraints)


__all__ = ["run_constraints"]
