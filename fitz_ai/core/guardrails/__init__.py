# fitz_ai/core/guardrails/__init__.py
"""
Epistemic Guardrails - Constraint system for epistemic correctness.

Guardrails inspect retrieved context and determine what conclusions are allowed.
They are orthogonal to retrieval (what's relevant) and generation (how to answer).

This is a core platform capability supporting epistemic honesty across all engines.

Usage:
    from fitz_ai.core.guardrails import (
        ConstraintResult,
        create_default_constraints,
        run_constraints,
    )
    from fitz_ai.core.governance import AnswerGovernor

    constraints = create_default_constraints(chat=chat_provider)
    results = run_constraints(query, chunks, constraints)
    decision = AnswerGovernor().decide(results)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ConstraintPlugin, ConstraintResult
from .plugins.answer_verification import AnswerVerificationConstraint
from .plugins.causal_attribution import CausalAttributionConstraint
from .plugins.conflict_aware import ConflictAwareConstraint
from .plugins.insufficient_evidence import InsufficientEvidenceConstraint
from .plugins.specific_info_type import SpecificInfoTypeConstraint
from .runner import run_constraints
from .semantic import SemanticMatcher
from .staged import ConstraintStage, StageContext, StagedConstraintPipeline, run_staged_constraints

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider


def create_default_constraints(
    chat: "ChatProvider | None" = None,
) -> list[ConstraintPlugin]:
    """
    Create the default constraint plugins using LLM-based detection.

    Args:
        chat: ChatProvider for LLM-based contradiction detection

    Returns:
        List of constraint plugins
    """
    return [
        # Empty context only - abstain if no chunks retrieved
        InsufficientEvidenceConstraint(chat=chat),
        # Keywords: "why" query + no "because" evidence
        CausalAttributionConstraint(),
        # LLM pairwise comparison: detect contradictions
        ConflictAwareConstraint(chat=chat),
        # LLM jury: verify chunks actually answer the query
        AnswerVerificationConstraint(chat=chat),
    ]


__all__ = [
    # Core types
    "ConstraintResult",
    "ConstraintPlugin",
    "SemanticMatcher",
    # Constraint implementations
    "ConflictAwareConstraint",
    "InsufficientEvidenceConstraint",
    "SpecificInfoTypeConstraint",
    "CausalAttributionConstraint",
    "AnswerVerificationConstraint",
    # Factory functions
    "create_default_constraints",
    # Runner
    "run_constraints",
    # Staged pipeline
    "StagedConstraintPipeline",
    "StageContext",
    "ConstraintStage",
    "run_staged_constraints",
]
