# fitz_ai/engines/fitz_rag/retrieval/multihop/__init__.py
"""
Multi-hop retrieval for complex queries.

Multi-hop retrieval iteratively retrieves evidence until sufficient information
is found to answer the query. This handles complex reasoning chains like:
"What does Sarah Chen's company's main competitor manufacture?"

Components:
- HopController: Orchestrates the retrieval loop
- EvidenceEvaluator: LLM judges if evidence is sufficient
- BridgeExtractor: LLM generates follow-up questions for missing info
"""

from __future__ import annotations

from fitz_ai.engines.fitz_rag.retrieval.multihop.controller import (
    HopController,
    HopMetadata,
    HopResult,
)
from fitz_ai.engines.fitz_rag.retrieval.multihop.evaluator import EvidenceEvaluator
from fitz_ai.engines.fitz_rag.retrieval.multihop.extractor import BridgeExtractor

__all__ = [
    "HopController",
    "HopMetadata",
    "HopResult",
    "EvidenceEvaluator",
    "BridgeExtractor",
]
