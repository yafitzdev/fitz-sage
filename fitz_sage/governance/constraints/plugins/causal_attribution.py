# fitz_sage/governance/constraints/plugins/causal_attribution.py
"""
Causal Attribution Constraint - Prevents implicit causality and speculation.

This constraint prevents the system from:
1. Synthesizing causal explanations when documents only describe outcomes
2. Making predictions when documents only contain historical data
3. Providing opinions/recommendations when documents only contain facts

It enforces: "Don't extrapolate beyond what the evidence explicitly states."

Requires an embedder to operate. Without one the constraint is a no-op —
provider presence IS the feature toggle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Sequence

from fitz_sage.governance.protocol import EvidenceItem
from fitz_sage.logging.logger import get_logger
from fitz_sage.logging.tags import PIPELINE

from ..base import ConstraintResult, FeatureSpec

if TYPE_CHECKING:
    from fitz_sage.governance.constraints.semantic import SemanticMatcher

logger = get_logger(__name__)


def _mentions_future_year(query: str) -> bool:
    """Check if query mentions a future year (e.g., 'in 2027' when current year is 2025).

    Structural check — year identifiers are format-based, not semantic.
    """
    current_year = datetime.now().year
    for year_str in re.findall(r"\b(20\d{2})\b", query):
        if int(year_str) > current_year:
            return True
    return False


@dataclass
class CausalAttributionConstraint:
    """
    Constraint that prevents implicit causal synthesis and speculation.

    Delegates to SemanticMatcher for language-agnostic detection of:
    - Causal queries: "why", "what caused" → need causal evidence
    - Predictive queries: "will", "next year" → need forecasts/projections
    - Opinion queries: "should we", "is it better" → almost always qualified
    - Speculative queries: "will succeed" → need explicit predictions

    If query asks for extrapolation but chunks only have facts, the constraint
    fires a "qualified" signal which governance resolves to TRUSTWORTHY
    (with caveats tracked via triggered_constraints).

    Requires embedder. Without one the constraint is a no-op.

    Attributes:
        enabled: Whether this constraint is active (default: True)
        embedder: Embedder instance for semantic detection
    """

    enabled: bool = True
    embedder: Any = field(default=None, repr=False, compare=False)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _semantic_matcher: "SemanticMatcher | None" = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.embedder is not None:
            from fitz_sage.governance.constraints.semantic import SemanticMatcher

            self._semantic_matcher = SemanticMatcher(
                embedder=lambda text: self.embedder.embed(text, task_type="query")
            )

    @property
    def name(self) -> str:
        return "causal_attribution"

    @staticmethod
    def feature_schema() -> list[FeatureSpec]:
        return [
            FeatureSpec("caa_fired", "bool", default=None),
            FeatureSpec("caa_query_type", "categorical", default="none"),
            FeatureSpec("caa_has_causal_evidence", "bool", default=None),
            FeatureSpec("caa_has_predictive_evidence", "bool", default=None),
        ]

    def apply(
        self,
        query: str,
        chunks: Sequence[EvidenceItem],
    ) -> ConstraintResult:
        """
        Check if uncertainty queries have sufficient evidence.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies confident answer if evidence is insufficient
        """
        if not self.enabled:
            return ConstraintResult.allow()

        # Empty chunks - defer to InsufficientEvidenceConstraint
        if not chunks:
            return ConstraintResult.allow()

        # Without embedder the constraint is a no-op
        if self._semantic_matcher is None:
            return ConstraintResult.allow()

        # Semantic query type classification
        is_uncertainty, query_type = self._semantic_matcher.is_uncertainty_query(query)

        # Structural augmentation: future year mentions always flag as predictive
        if not is_uncertainty and _mentions_future_year(query):
            is_uncertainty, query_type = True, "predictive"

        # Compute evidence signals — only raw content, NOT summaries.
        # LLM-generated summaries use causal language in a meta way which creates
        # false positives (e.g. "This describes X because...").
        has_causal = any(
            self._semantic_matcher.has_causal_language(chunk.content) for chunk in chunks
        )
        has_predictive = any(
            self._semantic_matcher.has_predictive_language(chunk.content) for chunk in chunks
        )

        caa_diag = {
            "caa_query_type": query_type,
            "caa_has_causal_evidence": has_causal,
            "caa_has_predictive_evidence": has_predictive,
        }

        if not is_uncertainty:
            logger.debug(f"{PIPELINE} CausalAttributionConstraint: not an uncertainty query")
            return ConstraintResult.allow(**caa_diag)

        if self._has_appropriate_evidence(query_type, has_causal, has_predictive):
            logger.debug(f"{PIPELINE} CausalAttributionConstraint: {query_type} evidence found")
            return ConstraintResult.allow(**caa_diag)

        logger.info(
            f"{PIPELINE} CausalAttributionConstraint: {query_type} query but no "
            f"supporting evidence"
        )
        return ConstraintResult.deny(
            reason=f"{query_type.capitalize()} query but no supporting evidence in context",
            signal="qualified",
            query_type=query_type,
            total_chunks=len(chunks),
            **caa_diag,
        )

    @staticmethod
    def _has_appropriate_evidence(query_type: str, has_causal: bool, has_predictive: bool) -> bool:
        """Check if evidence matches the query type."""
        if query_type == "causal":
            return has_causal
        if query_type in ("predictive", "speculative"):
            return has_predictive
        if query_type == "opinion":
            # Opinion queries almost never have definitive evidence
            return False
        return True


__all__ = ["CausalAttributionConstraint"]
