# fitz_ai/core/guardrails/plugins/causal_attribution.py
"""
Causal Attribution Constraint - Prevents implicit causality claims.

This constraint prevents the system from synthesizing causal explanations
when documents only describe outcomes without explicit causal language.

Uses simple keyword detection - no embeddings, no thresholds.
- Causal queries: "why", "what caused", "what led to", etc.
- Causal evidence: "because", "due to", "caused by", etc.

It enforces: "Don't invent causality that isn't explicitly stated."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult

logger = get_logger(__name__)


# Keywords that indicate a causal query
CAUSAL_QUERY_PATTERNS = (
    "why ",
    "why?",
    "what caused",
    "what led to",
    "what leads to",
    "explain why",
    "what's the reason",
    "what is the reason",
    "how come",
    "what made",
)

# Keywords that indicate causal evidence in text
CAUSAL_EVIDENCE_KEYWORDS = (
    "because",
    "due to",
    "caused by",
    "led to",
    "leads to",
    "as a result",
    "result of",
    "therefore",
    "thus",
    "consequently",
    "owing to",
    "reason is",
    "reason was",
    "the cause",
    "attributed to",
)


def _is_causal_query(query: str) -> bool:
    """Check if query is asking for causal explanation using keywords."""
    q = query.lower().strip()
    return any(pattern in q for pattern in CAUSAL_QUERY_PATTERNS)


def _has_causal_evidence(chunks: Sequence[Chunk]) -> bool:
    """Check if any chunk contains causal language using keywords."""
    for chunk in chunks:
        content = chunk.content.lower()
        if any(kw in content for kw in CAUSAL_EVIDENCE_KEYWORDS):
            return True
    return False


@dataclass
class CausalAttributionConstraint:
    """
    Constraint that prevents implicit causal synthesis.

    Uses simple keyword detection:
    - Detects causal queries: "why", "what caused", etc.
    - Detects causal evidence: "because", "due to", etc.

    If query asks "why" but chunks don't explain "because" → QUALIFIED.

    No embeddings, no thresholds - just keyword matching.

    Attributes:
        enabled: Whether this constraint is active (default: True)
    """

    enabled: bool = True
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def name(self) -> str:
        return "causal_attribution"

    def apply(
        self,
        query: str,
        chunks: Sequence[Chunk],
    ) -> ConstraintResult:
        """
        Check if causal queries have sufficient causal evidence.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies causal synthesis if evidence is insufficient
        """
        if not self.enabled:
            return ConstraintResult.allow()

        # Empty chunks - defer to InsufficientEvidenceConstraint
        if not chunks:
            return ConstraintResult.allow()

        # Check if query is asking for causal explanation
        if not _is_causal_query(query):
            logger.debug(f"{PIPELINE} CausalAttributionConstraint: not a causal query")
            return ConstraintResult.allow()

        # Check if chunks contain causal language
        if _has_causal_evidence(chunks):
            logger.debug(f"{PIPELINE} CausalAttributionConstraint: causal evidence found")
            return ConstraintResult.allow()

        # Causal query without causal evidence - deny
        logger.info(
            f"{PIPELINE} CausalAttributionConstraint: causal query but no causal evidence"
        )
        return ConstraintResult.deny(
            reason="Causal query but no causal evidence in context",
            signal="qualified",
            total_chunks=len(chunks),
        )


__all__ = ["CausalAttributionConstraint"]
