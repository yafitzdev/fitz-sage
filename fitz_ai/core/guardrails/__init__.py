# fitz_ai/core/guardrails/__init__.py
"""
Epistemic Guardrails - Constraint system for epistemic correctness.

Guardrails inspect retrieved context and determine what conclusions are allowed.
They are orthogonal to retrieval (what's relevant) and generation (how to answer).

This is a core platform capability supporting epistemic honesty across all engines.

## Two Modes

1. **LLM-based** (create_default_constraints):
   - Uses LLM for contradiction detection
   - Subject to model variance
   - Better for edge cases

2. **Deterministic** (create_deterministic_constraints):
   - Uses embeddings + regex for contradiction detection
   - No LLM variance - fully deterministic
   - Tunable thresholds
   - Explainable results

Usage:
    from fitz_ai.core.guardrails import (
        ConstraintResult,
        create_deterministic_constraints,
        run_constraints,
    )
    from fitz_ai.core.governance import AnswerGovernor

    # Deterministic constraints (recommended)
    constraints = create_deterministic_constraints(embedder=embed_func)

    # Run constraints and get governance decision
    results = run_constraints(query, chunks, constraints)
    decision = AnswerGovernor().decide(results)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .base import ConstraintPlugin, ConstraintResult
from .plugins.answer_verification import AnswerVerificationConstraint
from .plugins.causal_attribution import CausalAttributionConstraint
from .plugins.conflict_aware import ConflictAwareConstraint
from .plugins.deterministic_conflict import DeterministicConflictConstraint
from .plugins.governance_analyzer import GovernanceAnalyzer
from .plugins.insufficient_evidence import InsufficientEvidenceConstraint
from .plugins.specific_info_type import SpecificInfoTypeConstraint
from .runner import run_constraints
from .semantic import SemanticMatcher
from .staged import ConstraintStage, StagedConstraintPipeline, StageContext, run_staged_constraints

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider

    from .semantic import EmbedderFunc

# Type alias
EmbedderFunc = Callable[[str], list[float]]


def create_deterministic_constraints(
    embedder: EmbedderFunc,
    similarity_threshold: float = 0.6,
    relevance_threshold: float = 0.4,
) -> list[ConstraintPlugin]:
    """
    Create deterministic constraint plugins using embeddings + regex.

    NO LLM calls. Fully deterministic. Tunable thresholds.

    Pipeline:
    - Abstention: embedding similarity (query ↔ chunks) < relevance_threshold
    - Dispute: embedding similarity (chunk ↔ chunk) > similarity_threshold AND antonym pairs
    - Qualification: keyword-based causal detection (already deterministic)

    Args:
        embedder: Function to embed text into vectors
        similarity_threshold: Min similarity to consider "same topic" for contradiction (default: 0.6)
        relevance_threshold: Min similarity to consider chunks relevant to query (default: 0.4)

    Returns:
        List of deterministic constraint plugins
    """
    return [
        # Embedding-based relevance check
        InsufficientEvidenceConstraint(
            embedder=embedder,
            min_similarity=relevance_threshold,
        ),
        # Keywords: "why" query + no "because" evidence
        CausalAttributionConstraint(),
        # Embeddings + regex antonyms: no LLM variance
        DeterministicConflictConstraint(
            embedder=embedder,
            similarity_threshold=similarity_threshold,
        ),
    ]


def create_default_constraints(
    chat: "ChatProvider | None" = None,
    # Legacy parameters - kept for backwards compatibility but ignored
    semantic_matcher: SemanticMatcher | None = None,
    embedder: "EmbedderFunc | None" = None,
) -> list[ConstraintPlugin]:
    """
    Create the default constraint plugins using LLM-based detection.

    Note: Subject to LLM model variance. For deterministic results,
    use create_deterministic_constraints() instead.

    Args:
        chat: ChatProvider for LLM-based contradiction detection
        semantic_matcher: DEPRECATED - ignored
        embedder: DEPRECATED - ignored

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


def create_semantic_matcher(embedder: "EmbedderFunc") -> SemanticMatcher:
    """
    Create a SemanticMatcher with the given embedder function.

    DEPRECATED: Use create_deterministic_constraints() instead.

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
    "SemanticMatcher",
    # Constraint implementations
    "GovernanceAnalyzer",
    "ConflictAwareConstraint",
    "DeterministicConflictConstraint",
    "InsufficientEvidenceConstraint",
    "SpecificInfoTypeConstraint",
    "CausalAttributionConstraint",
    "AnswerVerificationConstraint",
    # Factory functions
    "create_deterministic_constraints",
    "create_default_constraints",
    "create_semantic_matcher",
    # Runner
    "run_constraints",
    # Staged pipeline
    "StagedConstraintPipeline",
    "StageContext",
    "ConstraintStage",
    "run_staged_constraints",
]
