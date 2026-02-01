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
        mode: The resolved epistemic posture (CONFIDENT/QUALIFIED/DISPUTED/ABSTAIN)
        triggered_constraints: Names of constraints that denied decisive answers
        reasons: Human-readable explanations from each triggered constraint
        signals: Raw signal strings from constraints (for debugging/logging)
    """

    mode: AnswerMode
    triggered_constraints: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    signals: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_confident(self) -> bool:
        """True if no constraints restricted the answer."""
        return self.mode == AnswerMode.CONFIDENT

    @property
    def should_include_caveats(self) -> bool:
        """True if the answer should include uncertainty language."""
        return self.mode in (AnswerMode.QUALIFIED, AnswerMode.DISPUTED)

    @property
    def user_explanation(self) -> str | None:
        """
        Explanation suitable for showing to the user.

        Returns None for CONFIDENT mode (no explanation needed).
        For other modes, returns a human-readable reason.
        """
        if self.mode == AnswerMode.CONFIDENT:
            return None

        if self.mode == AnswerMode.ABSTAIN:
            if self.reasons:
                return "; ".join(self.reasons)
            return "I don't have enough information to answer this question."

        if self.mode == AnswerMode.DISPUTED:
            if self.reasons:
                return f"Sources contain conflicting information: {'; '.join(self.reasons)}"
            return "Sources contain conflicting information."

        # QUALIFIED
        if self.reasons:
            return "; ".join(self.reasons)
        return "Answer provided with noted limitations."

    def to_dict(self) -> dict:
        """Serialize for logging/storage."""
        return {
            "mode": self.mode.value,
            "triggered_constraints": list(self.triggered_constraints),
            "reasons": list(self.reasons),
            "signals": list(self.signals),
        }

    @classmethod
    def confident(cls) -> "GovernanceDecision":
        """Factory for confident decisions (no constraints triggered)."""
        return cls(mode=AnswerMode.CONFIDENT)


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
    3. any denial → QUALIFIED (some limitation)
    4. all pass → CONFIDENT (no restrictions)
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
            return GovernanceDecision.confident()

        # Collect data from all constraint results
        triggered: list[str] = []
        reasons: list[str] = []
        signals: set[str] = set()

        for result in results:
            if not result.allow_decisive_answer:
                # Get constraint name from metadata or use "unknown"
                name = result.metadata.get("constraint_name", "unknown")
                triggered.append(name)

                if result.reason:
                    reasons.append(result.reason)
                if result.signal:
                    signals.add(result.signal)

        # Resolve mode using signal priority
        mode = self._resolve_mode(signals, has_denials=bool(triggered))

        return GovernanceDecision(
            mode=mode,
            triggered_constraints=tuple(triggered),
            reasons=tuple(reasons),
            signals=frozenset(signals),
        )

    def _resolve_mode(self, signals: set[str], has_denials: bool) -> AnswerMode:
        """
        Resolve answer mode from signals using priority order.

        Priority:
        1. "abstain" present → ABSTAIN
        2. "disputed" present → DISPUTED
        3. Any denials (even without signal) → QUALIFIED
        4. Otherwise → CONFIDENT
        """
        if "abstain" in signals:
            return AnswerMode.ABSTAIN
        if "disputed" in signals:
            return AnswerMode.DISPUTED
        if has_denials:
            return AnswerMode.QUALIFIED
        return AnswerMode.CONFIDENT


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
