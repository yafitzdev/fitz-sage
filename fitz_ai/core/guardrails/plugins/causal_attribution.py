# fitz_ai/core/guardrails/plugins/causal_attribution.py
"""
Causal Attribution Constraint - Prevents implicit causality claims.

This constraint prevents the system from synthesizing causal explanations
when documents only describe outcomes without explicit causal language.

Uses LLM-based analysis to detect:
- Causal queries (why, what caused, etc.)
- Whether evidence supports causal claims or just shows correlation
- Small samples, incomplete evidence, attribution errors

It enforces: "Don't invent causality that isn't explicitly stated."

This is NOT reasoning suppression. It's epistemic honesty enforcement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult
from ..semantic import SemanticMatcher

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider

logger = get_logger(__name__)


def _extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response, handling common issues."""
    text = text.strip()

    # Remove markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()

    # Try to find JSON object in the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


# =============================================================================
# Prompts
# =============================================================================

CAUSAL_ANALYSIS_PROMPT = """Analyze if this query asks for causal explanation and if the context provides sufficient causal evidence.

Query: {query}

Context:
{context}

Consider:
1. Is this a CAUSAL query? (asks "why", "what caused", "what led to", predictions needing causes)
2. Does context provide ACTUAL CAUSAL EVIDENCE or just:
   - Correlations/statistics without mechanism
   - Multiple potential causes (can't attribute to one)
   - Small sample sizes
   - Incomplete evidence
   - Observations without explanation

Respond with JSON only:
{{"is_causal_query": true/false, "has_causal_evidence": true/false, "reason": "brief explanation"}}

If NOT a causal query: {{"is_causal_query": false, "has_causal_evidence": true, "reason": "not asking why"}}
If causal query WITH sufficient causal evidence: {{"is_causal_query": true, "has_causal_evidence": true, "reason": "context explains cause"}}
If causal query WITHOUT causal evidence: {{"is_causal_query": true, "has_causal_evidence": false, "reason": "why insufficient"}}

JSON only:"""


# =============================================================================
# Constraint Implementation
# =============================================================================


@dataclass
class CausalAttributionConstraint:
    """
    Constraint that prevents implicit causal synthesis.

    When a query requests causal explanation (why, what caused, etc.),
    this constraint verifies that retrieved documents contain explicit
    causal evidence before allowing a causal answer.

    Uses LLM-based analysis to detect:
    - Whether query asks for causal explanation
    - Whether evidence provides actual causation or just correlation
    - Small samples, incomplete evidence, multiple confounding factors

    This prevents the LLM from inventing causal relationships that
    aren't explicitly stated in the evidence.

    Attributes:
        semantic_matcher: SemanticMatcher instance (for fallback)
        chat: Optional ChatProvider for LLM-based analysis
        enabled: Whether this constraint is active (default: True)
    """

    semantic_matcher: SemanticMatcher
    chat: "ChatProvider | None" = None
    enabled: bool = True
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def name(self) -> str:
        return "causal_attribution"

    def _analyze_causal_evidence(
        self, query: str, chunks: Sequence[Chunk]
    ) -> tuple[bool, bool, str]:
        """
        Use LLM to analyze if query is causal and if evidence supports it.

        Returns:
            Tuple of (is_causal_query, has_causal_evidence, reason)
        """
        if not self.chat:
            # Fallback to embedding-based detection
            is_causal = self.semantic_matcher.is_causal_query(query)
            has_evidence = self.semantic_matcher.count_causal_chunks(chunks) > 0
            return is_causal, has_evidence, "embedding-based fallback"

        # Format context for prompt
        context_texts = []
        for i, chunk in enumerate(chunks[:5]):  # Limit to 5 chunks
            context_texts.append(f"[{i+1}]: {chunk.content[:400]}")

        context_str = "\n\n".join(context_texts)
        prompt = CAUSAL_ANALYSIS_PROMPT.format(query=query, context=context_str)

        try:
            response = self.chat.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )

            logger.debug(f"{PIPELINE} CausalAttributionConstraint: LLM response: {response[:200]}")

            result = _extract_json(response)

            if result is None:
                logger.warning(
                    f"{PIPELINE} CausalAttributionConstraint: could not extract JSON from: {response[:100]}"
                )
                # Fallback to embedding-based
                is_causal = self.semantic_matcher.is_causal_query(query)
                has_evidence = self.semantic_matcher.count_causal_chunks(chunks) > 0
                return is_causal, has_evidence, "JSON parse fallback"

            is_causal = result.get("is_causal_query", False)
            has_evidence = result.get("has_causal_evidence", True)
            reason = result.get("reason", "")

            logger.debug(
                f"{PIPELINE} CausalAttributionConstraint: is_causal={is_causal}, "
                f"has_evidence={has_evidence}, reason={reason}"
            )

            return is_causal, has_evidence, reason

        except Exception as e:
            logger.warning(f"{PIPELINE} CausalAttributionConstraint: LLM call failed: {e}")
            # Fallback to embedding-based
            is_causal = self.semantic_matcher.is_causal_query(query)
            has_evidence = self.semantic_matcher.count_causal_chunks(chunks) > 0
            return is_causal, has_evidence, "exception fallback"

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

        # Analyze using LLM (or fallback to embeddings)
        is_causal_query, has_causal_evidence, reason = self._analyze_causal_evidence(
            query, chunks
        )

        # Not a causal query - allow
        if not is_causal_query:
            logger.debug(f"{PIPELINE} CausalAttributionConstraint: not a causal query, allowing")
            return ConstraintResult.allow()

        # Causal query with sufficient evidence - allow
        if has_causal_evidence:
            logger.debug(f"{PIPELINE} CausalAttributionConstraint: causal evidence found, allowing")
            return ConstraintResult.allow()

        # Causal query without sufficient evidence - deny
        logger.info(
            f"{PIPELINE} CausalAttributionConstraint: causal query but insufficient "
            f"causal evidence: {reason}"
        )

        return ConstraintResult.deny(
            reason=f"Insufficient causal evidence: {reason}",
            signal="qualified",  # Not abstain - we have evidence, just not causal
            total_chunks=len(chunks),
        )


__all__ = ["CausalAttributionConstraint"]
