# fitz_ai/core/guardrails/plugins/causal_attribution.py
"""
Causal Attribution Constraint - Prevents implicit causality and speculation.

This constraint prevents the system from:
1. Synthesizing causal explanations when documents only describe outcomes
2. Making predictions when documents only contain historical data
3. Providing opinions/recommendations when documents only contain facts

Uses simple keyword detection - no embeddings, no thresholds.

It enforces: "Don't extrapolate beyond what the evidence explicitly states."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
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

# Keywords that indicate predictive queries (future-oriented)
PREDICTIVE_QUERY_PATTERNS = (
    "will ",
    "will?",
    "what will",
    "how will",
    "next week",
    "next month",
    "next year",
    "next quarter",
    "by year end",
    "going to be",
    "going to happen",
    "be like in",
    "forecast",
    "predict",
    "projection",
    "outlook",
    "expect",
    "estimate for",
    "anticipated",
)

# Keywords that indicate opinion/judgment queries
OPINION_QUERY_PATTERNS = (
    "should we",
    "should i",
    "is it better",
    "is it best",
    "which is better",
    "which is best",
    "what's the best",
    "what is the best",
    "is it worth",
    "is this effective",
    "is this adequate",
    "based on",  # "Based on X, should we..."
    " better for ",  # "Is X better for Y?"
    " better than ",  # "Is X better than Y?"
    " best for ",  # "What is best for X?"
)

# Regex pattern for "recommend" with word boundary to avoid matching "recommendation"
_RECOMMEND_RE = re.compile(r"\brecommend(s|ed|ing)?\b", re.IGNORECASE)

# Exclusion patterns: queries that use opinion keywords but are factual/comparative
_OPINION_EXCLUSIONS = (
    # "Should we use X or Y" → factual comparison, not opinion-seeking
    re.compile(r"^should\s+(we|i)\s+use\s+.+\s+or\s+", re.IGNORECASE),
    # "What/which ... should I ... to/for" → factual best-practice question
    re.compile(r"^(what|which|how)\s+.+\s+should\s+(i|we)\s+.+\s+(to|for)\b", re.IGNORECASE),
)

# Keywords that indicate speculative queries
SPECULATIVE_QUERY_PATTERNS = (
    "what percentage will",
    "how many will",
    "what impact will",
    "will succeed",
    "will fail",
    "will be approved",
    "will become",
    "be successful",
    "be mainstream",
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

# Keywords that indicate predictive/forward-looking evidence
PREDICTIVE_EVIDENCE_KEYWORDS = (
    "forecast",
    "projected",
    "prediction",
    "expected to",
    "likely to",
    "will likely",
    "estimated to reach",
    "anticipated",
    "outlook",
)


def _is_causal_query(query: str) -> bool:
    """Check if query is asking for causal explanation using keywords."""
    q = query.lower().strip()
    return any(pattern in q for pattern in CAUSAL_QUERY_PATTERNS)


def _mentions_future_year(query: str) -> bool:
    """Check if query mentions a future year (e.g., 'in 2026' when current year is 2025)."""
    current_year = datetime.now().year
    # Match patterns like "in 2025", "by 2026", "2027 forecast"
    year_pattern = r"\b(20\d{2})\b"
    matches = re.findall(year_pattern, query)
    for year_str in matches:
        year = int(year_str)
        if year > current_year:
            return True
    return False


def _is_predictive_query(query: str) -> bool:
    """Check if query is asking for predictions about the future."""
    q = query.lower().strip()
    if any(pattern in q for pattern in PREDICTIVE_QUERY_PATTERNS):
        return True
    # Also check for future year mentions
    return _mentions_future_year(query)


def _is_opinion_query(query: str) -> bool:
    """Check if query is asking for opinions, recommendations, or judgments."""
    q = query.lower().strip()

    # Check exclusions first — these look like opinion queries but are factual
    if any(exc.search(q) for exc in _OPINION_EXCLUSIONS):
        return False

    if any(pattern in q for pattern in OPINION_QUERY_PATTERNS):
        return True
    # Word-boundary check for "recommend" (avoids "recommendation")
    if _RECOMMEND_RE.search(q):
        return True
    # Regex patterns for comparative questions: "Is X better", "Which X is better"
    comparative_patterns = [
        r"^is\s+\w+.*\s+better",  # "Is remote work better..."
        r"^which\s+\w+.*\s+better",  # "Which framework is better..."
        r"^which\s+\w+.*\s+best",  # "Which option is best..."
        r"\bbetter\s*\?",  # ends with "better?"
        r"\bbest\s*\?",  # ends with "best?"
    ]
    return any(re.search(p, q) for p in comparative_patterns)


def _is_speculative_query(query: str) -> bool:
    """Check if query requires speculation beyond available facts."""
    q = query.lower().strip()
    return any(pattern in q for pattern in SPECULATIVE_QUERY_PATTERNS)


def _is_uncertainty_query(query: str) -> tuple[bool, str]:
    """
    Check if query requires qualification due to inherent uncertainty.

    Returns (is_uncertainty_query, query_type).
    """
    if _is_causal_query(query):
        return True, "causal"
    if _is_predictive_query(query):
        return True, "predictive"
    if _is_opinion_query(query):
        return True, "opinion"
    if _is_speculative_query(query):
        return True, "speculative"
    return False, "none"


def _has_causal_evidence(chunks: Sequence[Chunk]) -> bool:
    """Check if any chunk contains causal language using keywords.

    Only checks raw content - NOT summaries. LLM-generated summaries often
    use causal language in a meta way ("This describes X because...") which
    creates false positives for evidence detection.
    """
    for chunk in chunks:
        content = chunk.content.lower()
        if any(kw in content for kw in CAUSAL_EVIDENCE_KEYWORDS):
            return True

    return False


def _has_predictive_evidence(chunks: Sequence[Chunk]) -> bool:
    """Check if any chunk contains forward-looking/predictive language.

    Only checks raw content - NOT summaries (same rationale as causal evidence).
    """
    for chunk in chunks:
        content = chunk.content.lower()
        if any(kw in content for kw in PREDICTIVE_EVIDENCE_KEYWORDS):
            return True

    return False


def _has_appropriate_evidence(query_type: str, chunks: Sequence[Chunk]) -> bool:
    """Check if chunks have evidence appropriate to the query type."""
    if query_type == "causal":
        return _has_causal_evidence(chunks)
    elif query_type in ("predictive", "speculative"):
        return _has_predictive_evidence(chunks)
    elif query_type == "opinion":
        # Opinion queries almost never have definitive evidence
        # Would need explicit recommendations in the text
        return False
    return True


@dataclass
class CausalAttributionConstraint:
    """
    Constraint that prevents implicit causal synthesis and speculation.

    Uses simple keyword detection to identify queries that require qualification:
    - Causal queries: "why", "what caused" → need "because", "due to"
    - Predictive queries: "will", "next year" → need forecasts/projections
    - Opinion queries: "should we", "is it better" → almost always qualified
    - Speculative queries: "will succeed" → need explicit predictions

    If query asks for extrapolation but chunks only have facts → QUALIFIED.

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

        # Check if query requires qualification
        is_uncertainty, query_type = _is_uncertainty_query(query)

        # Compute evidence signals for classifier feature extraction
        has_causal = _has_causal_evidence(chunks)
        has_predictive = _has_predictive_evidence(chunks)
        caa_diag = {
            "caa_query_type": query_type,
            "caa_has_causal_evidence": has_causal,
            "caa_has_predictive_evidence": has_predictive,
        }

        if not is_uncertainty:
            logger.debug(f"{PIPELINE} CausalAttributionConstraint: not an uncertainty query")
            return ConstraintResult.allow(**caa_diag)

        # Check if chunks have appropriate evidence
        if _has_appropriate_evidence(query_type, chunks):
            logger.debug(f"{PIPELINE} CausalAttributionConstraint: {query_type} evidence found")
            return ConstraintResult.allow(**caa_diag)

        # Uncertainty query without appropriate evidence - deny
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


__all__ = ["CausalAttributionConstraint"]
