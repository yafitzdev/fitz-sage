# fitz_ai/core/guardrails/plugins/__init__.py
"""
Constraint Plugins - Individual guardrail implementations.

Default plugins:
- ConflictAwareConstraint: detects contradicting sources
- InsufficientEvidenceConstraint: detects missing evidence
- CausalAttributionConstraint: prevents implicit causality claims
"""

from .causal_attribution import CausalAttributionConstraint
from .conflict_aware import ConflictAwareConstraint
from .insufficient_evidence import InsufficientEvidenceConstraint

__all__ = [
    "ConflictAwareConstraint",
    "InsufficientEvidenceConstraint",
    "CausalAttributionConstraint",
]
