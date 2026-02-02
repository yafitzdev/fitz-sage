# fitz_ai/core/guardrails/__init__.py
"""
Epistemic Guardrails - Constraint system for epistemic correctness.

Guardrails inspect retrieved context and determine what conclusions are allowed.
They are orthogonal to retrieval (what's relevant) and generation (how to answer).

This is a core platform capability supporting epistemic honesty across all engines.

## Simple Architecture

All constraints use simple LLM YES/NO classification or keyword matching.
No embeddings, no thresholds - just straightforward logic.

Default constraints:
- InsufficientEvidenceConstraint: LLM YES/NO "Is this relevant?"
- CausalAttributionConstraint: Keywords for "why" queries and "because" evidence
- ConflictAwareConstraint: LLM YES/NO stance per chunk, detect YES vs NO

Usage:
    from fitz_ai.core.guardrails import (
        ConstraintResult,
        ConstraintPlugin,
        create_default_constraints,
        run_constraints,
    )
    from fitz_ai.core.governance import AnswerGovernor

    # Get default constraints (requires chat for LLM calls)
    constraints = create_default_constraints(chat=fast_chat)

    # Run constraints and get governance decision
    results = run_constraints(query, chunks, constraints)
    decision = AnswerGovernor().decide(results)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ConstraintPlugin, ConstraintResult
from .plugins.causal_attribution import CausalAttributionConstraint
from .plugins.conflict_aware import ConflictAwareConstraint
from .plugins.governance_analyzer import GovernanceAnalyzer
from .plugins.insufficient_evidence import InsufficientEvidenceConstraint
from .runner import run_constraints
from .semantic import SemanticMatcher

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider

    from .semantic import EmbedderFunc


def create_default_constraints(
    chat: "ChatProvider | None" = None,
    # Legacy parameters - kept for backwards compatibility but ignored
    semantic_matcher: SemanticMatcher | None = None,
    embedder: "EmbedderFunc | None" = None,
) -> list[ConstraintPlugin]:
    """
    Create the default constraint plugins using simple architecture.

    All constraints use either:
    - Simple LLM YES/NO classification (relevance, contradiction)
    - Keyword matching (causal query/evidence detection)

    No embeddings, no thresholds.

    Args:
        chat: ChatProvider for LLM-based checks (required for relevance and contradiction)
        semantic_matcher: DEPRECATED - ignored
        embedder: DEPRECATED - ignored

    Returns:
        List of constraint plugins
    """
    return [
        # Empty context only - abstain if no chunks retrieved
        InsufficientEvidenceConstraint(),
        # Keywords: "why" query + no "because" evidence
        CausalAttributionConstraint(),
        # LLM YES/NO per chunk: detect YES vs NO stance
        ConflictAwareConstraint(chat=chat),
    ]


def create_semantic_matcher(embedder: "EmbedderFunc") -> SemanticMatcher:
    """
    Create a SemanticMatcher with the given embedder function.

    DEPRECATED: SemanticMatcher is no longer used by default constraints.
    Kept for backwards compatibility.

    Args:
        embedder: Function that converts text to embedding vector.

    Returns:
        Configured SemanticMatcher instance
    """
    return SemanticMatcher(embedder=embedder)


__all__ = [
    # Core types
    "ConstraintResult",
    "ConstraintPlugin",
    "SemanticMatcher",  # Kept for backwards compatibility
    # Constraint implementations
    "GovernanceAnalyzer",  # Legacy unified LLM-based
    "ConflictAwareConstraint",
    "InsufficientEvidenceConstraint",
    "CausalAttributionConstraint",
    # Functions
    "run_constraints",
    "create_default_constraints",
    "create_semantic_matcher",
]
