# fitz_ai/core/guardrails/plugins/conflict_aware.py
"""
Conflict-Aware Constraint - Default guardrail for contradiction detection.

This constraint detects conflicting claims in retrieved chunks using
simple LLM-based YES/NO classification per chunk.

The key insight: contradiction detection is the ONE thing keywords can't do.
A chunk saying "revenue increased" and one saying "revenue decreased" have
similar keywords but opposite stances.

This uses a simple per-chunk stance classification:
1. For each chunk: "Does this answer the query with YES or NO?"
2. If some chunks say YES and others say NO → contradiction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider

logger = get_logger(__name__)


STANCE_PROMPT = """Does this text answer the question with YES or NO?
If unclear or not applicable, say UNCLEAR.

Question: {query}
Text: {text}

Reply with ONLY one word: YES, NO, or UNCLEAR"""


# Keywords that indicate user is asking for conflict resolution
RESOLUTION_KEYWORDS = (
    "authoritative",
    "which source",
    "trust",
    "resolve",
    "reconcile",
    "correct version",
    "which is right",
    "which is correct",
)


def _is_resolution_query(query: str) -> bool:
    """Check if query is asking to resolve conflicts (keyword-based)."""
    q = query.lower()
    return any(kw in q for kw in RESOLUTION_KEYWORDS)


@dataclass
class ConflictAwareConstraint:
    """
    Constraint that detects conflicting claims using simple YES/NO classification.

    When retrieved chunks contain mutually exclusive claims (e.g., one says
    "revenue increased" and another says "revenue decreased"), this constraint
    prevents the system from confidently asserting either.

    Uses simple per-chunk stance detection:
    - Ask LLM: "Does this chunk say YES or NO to the query?"
    - If some say YES and some say NO → contradiction

    Attributes:
        chat: ChatProvider for stance classification
        enabled: Whether this constraint is active (default: True)
    """

    chat: "ChatProvider | None" = None
    enabled: bool = True
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def name(self) -> str:
        return "conflict_aware"

    def _get_chunk_stance(self, query: str, chunk: Chunk) -> str:
        """
        Get chunk's stance on the query: YES, NO, or UNCLEAR.

        Uses a simple single-word response format that fast models handle well.
        """
        if not self.chat:
            return "UNCLEAR"

        # Truncate chunk content to keep prompt short
        text = chunk.content[:500] if len(chunk.content) > 500 else chunk.content
        prompt = STANCE_PROMPT.format(query=query, text=text)

        try:
            response = self.chat.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

            # Parse single-word response
            word = response.strip().upper()

            if "YES" in word:
                return "YES"
            elif "NO" in word:
                return "NO"
            return "UNCLEAR"

        except Exception as e:
            logger.warning(f"{PIPELINE} ConflictAwareConstraint: stance check failed: {e}")
            return "UNCLEAR"

    def apply(
        self,
        query: str,
        chunks: Sequence[Chunk],
    ) -> ConstraintResult:
        """
        Check for conflicting claims in retrieved chunks.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies decisive answer if conflicts detected
        """
        if not self.enabled:
            return ConstraintResult.allow()

        if not chunks:
            return ConstraintResult.allow()

        # Need at least 2 chunks to have a conflict
        if len(chunks) < 2:
            return ConstraintResult.allow()

        # If query explicitly asks for resolution, allow decisive answer
        if _is_resolution_query(query):
            logger.debug(f"{PIPELINE} ConflictAwareConstraint: resolution query detected, allowing")
            return ConstraintResult.allow()

        # Skip conflict detection if no LLM available
        if not self.chat:
            logger.debug(
                f"{PIPELINE} ConflictAwareConstraint: no chat provider, skipping conflict detection"
            )
            return ConstraintResult.allow()

        # Get stance for each chunk (limit to 5 for efficiency)
        chunks_to_check = list(chunks[:5])
        stances = [self._get_chunk_stance(query, c) for c in chunks_to_check]

        logger.debug(f"{PIPELINE} ConflictAwareConstraint: stances={stances}")

        # Check for contradiction: some YES and some NO
        has_yes = "YES" in stances
        has_no = "NO" in stances

        if has_yes and has_no:
            yes_count = stances.count("YES")
            no_count = stances.count("NO")
            logger.info(
                f"{PIPELINE} ConflictAwareConstraint: contradiction detected "
                f"({yes_count} YES, {no_count} NO)"
            )
            return ConstraintResult.deny(
                reason=f"Conflicting stances: {yes_count} say YES, {no_count} say NO",
                signal="disputed",
            )

        logger.debug(f"{PIPELINE} ConflictAwareConstraint: no contradiction detected")
        return ConstraintResult.allow()


__all__ = ["ConflictAwareConstraint"]
