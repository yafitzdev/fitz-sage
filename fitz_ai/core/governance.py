# fitz_ai/core/governance.py
"""
Answer Governance - Determines epistemic posture for answers.

Governance answers: "Are we allowed to answer, and how confidently?"

This is the decision layer between constraints (which detect problems)
and generation (which produces the answer). It is:
- Deterministic: same inputs → same outputs
- Explainable: decisions include reasons
- Observable: decisions can be logged and evaluated

The AnswerGovernor consumes constraint results and produces a
GovernanceDecision that controls answer generation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

from fitz_ai.core.answer_mode import AnswerMode

if TYPE_CHECKING:
    from fitz_ai.core.guardrails.base import ConstraintResult


@dataclass(frozen=True)
class GovernanceLog:
    """
    Structured log entry for a governance decision.

    Captures everything needed to understand, debug, and analyze governance
    decisions over time. Designed for PostgreSQL storage and observability.

    Attributes:
        timestamp: When the decision was made (UTC)
        query_hash: SHA256 of normalized query for deduplication
        query_text: Original query text (stored for debugging, not used in hash)
        mode: The resolved AnswerMode value
        triggered_constraints: Names of constraints that denied decisive answers
        signals: Raw signal strings from constraints (abstain, disputed, etc.)
        reasons: Human-readable explanations from triggered constraints
        chunk_count: Number of retrieved chunks available for the decision
        collection: The collection being queried
        latency_ms: Time to make governance decision (optional)
        pipeline_version: Version of the pipeline for tracking regressions
    """

    timestamp: datetime
    query_hash: str
    query_text: str | None
    mode: str
    triggered_constraints: tuple[str, ...]
    signals: tuple[str, ...]
    reasons: tuple[str, ...]
    chunk_count: int
    collection: str
    latency_ms: float | None = None
    pipeline_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON/logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "query_hash": self.query_hash,
            "query_text": self.query_text,
            "mode": self.mode,
            "triggered_constraints": list(self.triggered_constraints),
            "signals": list(self.signals),
            "reasons": list(self.reasons),
            "chunk_count": self.chunk_count,
            "collection": self.collection,
            "latency_ms": self.latency_ms,
            "pipeline_version": self.pipeline_version,
        }

    @classmethod
    def from_decision(
        cls,
        decision: "GovernanceDecision",
        query_hash: str,
        query_text: str | None,
        chunk_count: int,
        collection: str,
        latency_ms: float | None = None,
        pipeline_version: str | None = None,
    ) -> "GovernanceLog":
        """
        Create a GovernanceLog from a GovernanceDecision.

        Args:
            decision: The governance decision to log
            query_hash: Pre-computed hash of the query
            query_text: Original query text (for debugging)
            chunk_count: Number of chunks available for the decision
            collection: Collection being queried
            latency_ms: Time to make governance decision
            pipeline_version: Version of the pipeline

        Returns:
            GovernanceLog ready for storage
        """
        return cls(
            timestamp=datetime.now(timezone.utc),
            query_hash=query_hash,
            query_text=query_text,
            mode=decision.mode.value,
            triggered_constraints=decision.triggered_constraints,
            signals=tuple(decision.signals),
            reasons=decision.reasons,
            chunk_count=chunk_count,
            collection=collection,
            latency_ms=latency_ms,
            pipeline_version=pipeline_version,
        )

    @staticmethod
    def hash_query(query: str) -> str:
        """
        Normalize and hash a query for deduplication.

        Normalization: lowercase, strip whitespace.
        Hash: SHA256 truncated to 64 chars (256 bits).

        Args:
            query: The query string to hash

        Returns:
            64-character hex hash
        """
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass(frozen=True)
class GovernanceDecision:
    """
    The output of the governance layer.

    Bundles the resolved answer mode with the evidence that led to it.
    This object is passed to generation and can be logged for observability.

    Attributes:
        mode: The resolved epistemic posture (TRUSTWORTHY/DISPUTED/ABSTAIN)
        triggered_constraints: Names of constraints that denied decisive answers
        reasons: Human-readable explanations from each triggered constraint
        signals: Raw signal strings from constraints (for debugging/logging)
    """

    mode: AnswerMode
    triggered_constraints: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    signals: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_trustworthy(self) -> bool:
        """True if evidence supports answering."""
        return self.mode == AnswerMode.TRUSTWORTHY

    @property
    def should_include_caveats(self) -> bool:
        """True if the answer should include uncertainty language."""
        return self.mode == AnswerMode.DISPUTED

    @property
    def user_explanation(self) -> str | None:
        """
        Explanation suitable for showing to the user.

        Returns None for TRUSTWORTHY mode (no explanation needed).
        For other modes, returns a human-readable reason.
        """
        if self.mode == AnswerMode.TRUSTWORTHY:
            return None

        if self.mode == AnswerMode.ABSTAIN:
            if self.reasons:
                return "; ".join(self.reasons)
            return "I don't have enough information to answer this question."

        if self.mode == AnswerMode.DISPUTED:
            if self.reasons:
                return f"Sources contain conflicting information: {'; '.join(self.reasons)}"
            return "Sources contain conflicting information."

        return None

    def to_dict(self) -> dict:
        """Serialize for logging/storage."""
        return {
            "mode": self.mode.value,
            "triggered_constraints": list(self.triggered_constraints),
            "reasons": list(self.reasons),
            "signals": list(self.signals),
        }

    @classmethod
    def trustworthy(cls) -> "GovernanceDecision":
        """Factory for trustworthy decisions (no constraints triggered)."""
        return cls(mode=AnswerMode.TRUSTWORTHY)


class AnswerGovernor:
    """
    Resolves constraint results into a governance decision.

    The governor is intentionally simple and deterministic:
    - No LLM calls
    - No heuristics
    - No configuration
    - Pure function from constraint results to decision

    Signal priority (highest to lowest):
    1. "abstain" → ABSTAIN (insufficient evidence)
    2. "disputed" → DISPUTED (conflicting sources)
    3. all else → TRUSTWORTHY (evidence supports answering)

    Context-aware resolution:
    When InsufficientEvidenceConstraint fires, dispute signals are
    subordinated because contradictions in irrelevant or partially-relevant
    content are noise, not meaningful disagreement.
    """

    def decide(self, results: Sequence["ConstraintResult"]) -> GovernanceDecision:
        """
        Produce a governance decision from constraint results.

        Args:
            results: Individual ConstraintResult from each constraint plugin

        Returns:
            GovernanceDecision with resolved mode and explanations
        """
        if not results:
            return GovernanceDecision.trustworthy()

        # Collect data from all constraint results
        triggered: list[str] = []
        reasons: list[str] = []
        signals: set[str] = set()
        constraint_signals: dict[str, str | None] = {}
        constraint_metadata: dict[str, dict] = {}

        for result in results:
            if not result.allow_decisive_answer:
                # Get constraint name from metadata or use "unknown"
                name = result.metadata.get("constraint_name", "unknown")
                triggered.append(name)

                if result.reason:
                    reasons.append(result.reason)
                if result.signal:
                    signals.add(result.signal)
                constraint_signals[name] = result.signal
                constraint_metadata[name] = result.metadata

        # Resolve mode using signal priority with context-aware resolution
        mode = self._resolve_mode(
            signals, constraint_signals, constraint_metadata, has_denials=bool(triggered)
        )

        return GovernanceDecision(
            mode=mode,
            triggered_constraints=tuple(triggered),
            reasons=tuple(reasons),
            signals=frozenset(signals),
        )

    # Similarity threshold below which IE's "qualified" signal indicates
    # content is only loosely related, making dispute signals unreliable.
    # Above this threshold, content is genuinely related and disputes may
    # reflect real contradictions.
    _IE_LOW_RELEVANCE_THRESHOLD = 0.70

    def _resolve_mode(
        self,
        signals: set[str],
        constraint_signals: dict[str, str | None],
        constraint_metadata: dict[str, dict],
        has_denials: bool,
    ) -> AnswerMode:
        """
        Resolve answer mode from signals using priority order with
        context-aware dispute subordination.

        Priority:
        1. "abstain" present → ABSTAIN (includes dispute subordination)
        2. "disputed" present → DISPUTED (unless subordinated)
        3. All else → TRUSTWORTHY

        Dispute subordination:
        When insufficient_evidence fires abstain, disputes are always
        subordinated. When IE fires qualified with LOW similarity (below
        _IE_LOW_RELEVANCE_THRESHOLD), the content is only loosely related
        and disputes are noise. When IE fires qualified with HIGH
        similarity, the content is genuinely related and disputes may
        reflect real contradictions -- these are NOT subordinated.
        """
        # Check for explicit abstain from insufficient_evidence even when
        # the abstain signal might be overridden by other processing.
        # This ensures IE's abstain always takes priority over disputes.
        ie_signal = constraint_signals.get("insufficient_evidence")
        if ie_signal == "abstain":
            return AnswerMode.ABSTAIN

        if "abstain" in signals:
            return AnswerMode.ABSTAIN

        if "disputed" in signals:
            # When IE fires qualified, check similarity to determine if
            # the content is genuinely related (disputes meaningful) or
            # only loosely related (disputes are noise).
            if ie_signal == "qualified":
                ie_meta = constraint_metadata.get("insufficient_evidence", {})
                ie_similarity = ie_meta.get("max_similarity", 1.0)
                if ie_similarity < self._IE_LOW_RELEVANCE_THRESHOLD:
                    # Low similarity + IE qualified = content loosely related.
                    # Disputes in loosely-related content are noise.
                    return AnswerMode.TRUSTWORTHY
                # High similarity + IE qualified = content genuinely related
                # but missing specific info. Disputes may be meaningful.
                # Fall through to normal dispute resolution below.

            return AnswerMode.DISPUTED

        return AnswerMode.TRUSTWORTHY


# Module-level instance for convenience
_default_governor = AnswerGovernor()


def decide_answer_mode(results: Sequence["ConstraintResult"]) -> GovernanceDecision:
    """
    Convenience function using the default governor.

    Args:
        results: Constraint results to evaluate

    Returns:
        GovernanceDecision with resolved mode
    """
    return _default_governor.decide(results)


__all__ = [
    "GovernanceLog",
    "GovernanceDecision",
    "AnswerGovernor",
    "decide_answer_mode",
]
