# fitz_ai/core/guardrails/runner.py
"""
Constraint Runner - Applies constraint plugins to retrieved context.

Returns individual results from each constraint (not combined).
Signal aggregation happens in the governance layer, not here.
"""

from __future__ import annotations

from typing import Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from .base import ConstraintPlugin, ConstraintResult

logger = get_logger(__name__)


def run_constraints(
    query: str,
    chunks: Sequence[Chunk],
    constraints: Sequence[ConstraintPlugin],
) -> list[ConstraintResult]:
    """
    Apply all constraints and return individual results.

    Each constraint's result is preserved separately so signals
    are not lost during aggregation. Signal resolution happens
    in the governance layer via AnswerGovernor.decide().

    Args:
        query: The user's question
        chunks: Retrieved chunks (post-retrieval, pre-generation)
        constraints: List of constraint plugins to apply

    Returns:
        List of individual ConstraintResult objects (one per constraint)
    """
    if not constraints:
        return []

    logger.debug(f"{PIPELINE} Running {len(constraints)} constraint(s)")

    results: list[ConstraintResult] = []

    for constraint in constraints:
        try:
            result = constraint.apply(query, chunks)

            # Inject constraint name into metadata for traceability
            if not result.allow_decisive_answer:
                # Create new result with constraint name in metadata
                metadata = dict(result.metadata)
                metadata["constraint_name"] = constraint.name
                result = ConstraintResult(
                    allow_decisive_answer=result.allow_decisive_answer,
                    reason=result.reason,
                    signal=result.signal,
                    metadata=metadata,
                )
                logger.info(
                    f"{PIPELINE} Constraint '{constraint.name}' denied: {result.reason}"
                )
            else:
                logger.debug(f"{PIPELINE} Constraint '{constraint.name}' passed")

            results.append(result)

        except Exception as e:
            # Fail-safe: if constraint crashes, log and skip
            # Do NOT block the answer due to constraint errors
            logger.warning(
                f"{PIPELINE} Constraint '{constraint.name}' raised exception: {e}"
            )
            continue

    denied_count = sum(1 for r in results if not r.allow_decisive_answer)
    logger.debug(f"{PIPELINE} Constraints complete: {denied_count} denied")

    return results


__all__ = ["run_constraints"]
