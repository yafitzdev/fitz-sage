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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from fitz_ai.core.answer_mode import AnswerMode

if TYPE_CHECKING:
    from fitz_ai.core.guardrails.base import ConstraintResult


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
    "GovernanceDecision",
    "AnswerGovernor",
    "decide_answer_mode",
]
