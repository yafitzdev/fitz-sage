# fitz_ai/governance/__init__.py
"""
Epistemic Governance - Shared constraint system for epistemic correctness.

Guardrails inspect retrieved context and determine what conclusions are allowed.
They are orthogonal to retrieval (what's relevant) and generation (how to answer).

Usage:
    from fitz_ai.governance import (
        ConstraintResult,
        create_default_constraints,
        run_constraints,
        AnswerGovernor,
    )

    constraints = create_default_constraints(chat=chat_provider)
    results = run_constraints(query, chunks, constraints)
    decision = AnswerGovernor().decide(results)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constraints.base import ConstraintPlugin, ConstraintResult
from .constraints.plugins.answer_verification import AnswerVerificationConstraint
from .constraints.plugins.causal_attribution import CausalAttributionConstraint
from .constraints.plugins.conflict_aware import ConflictAwareConstraint
from .constraints.plugins.insufficient_evidence import InsufficientEvidenceConstraint
from .constraints.plugins.specific_info_type import SpecificInfoTypeConstraint
from .constraints.runner import run_constraints
from .constraints.semantic import SemanticMatcher
from .constraints.staged import (
    ConstraintStage,
    StageContext,
    StagedConstraintPipeline,
    run_staged_constraints,
)
from .governor import AnswerGovernor, GovernanceDecision, GovernanceLog, decide_answer_mode
from .protocol import EvidenceItem

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider, Embedder


def create_default_constraints(
    chat: "ChatProvider | None" = None,
    embedder: "Embedder | None" = None,
) -> list[ConstraintPlugin]:
    """
    Create the default constraint plugins using LLM-based detection.

    Args:
        chat: ChatProvider for LLM-based contradiction detection
        embedder: Embedder for IE similarity checks (enables entity/aspect detection)

    Returns:
        List of constraint plugins
    """
    return [
        # Embedding similarity + entity matching for relevance detection
        InsufficientEvidenceConstraint(chat=chat, embedder=embedder),
        # Keywords: "why" query + no "because" evidence
        CausalAttributionConstraint(),
        # LLM pairwise comparison: detect contradictions
        ConflictAwareConstraint(chat=chat),
        # LLM jury: verify chunks actually answer the query
        AnswerVerificationConstraint(chat=chat),
    ]


__all__ = [
    # Protocol
    "EvidenceItem",
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
    # Governance
    "AnswerGovernor",
    "GovernanceDecision",
    "GovernanceLog",
    "decide_answer_mode",
]
