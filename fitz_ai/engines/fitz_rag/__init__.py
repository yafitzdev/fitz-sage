# fitz_ai/engines/fitz_rag/__init__.py
"""
Fitz RAG Engine - Retrieval-Augmented Generation implementation.

This package implements the Fitz RAG paradigm as a KnowledgeEngine.

Public API:
    - FitzRagEngine: The engine implementation
    - run_fitz_rag: Simple entry point for one-off queries
    - create_fitz_rag_engine: Factory for creating reusable engines

Examples:
    Simple usage:
    >>> from fitz_ai.engines.fitz_rag import run_fitz_rag
    >>> answer = run_fitz_rag("What is quantum computing?")
    >>> print(answer.text)

    Advanced usage:
    >>> from fitz_ai.engines.fitz_rag import FitzRagEngine
    >>> from fitz_ai.config import load_engine_config
    >>> from fitz_ai.core import Query
    >>>
    >>> config = load_engine_config("fitz_rag")
    >>> engine = FitzRagEngine(config)
    >>>
    >>> query = Query(text="Explain entanglement")
    >>> answer = engine.answer(query)
"""

from .engine import FitzRagEngine
from .runtime import create_fitz_rag_engine, run, run_fitz_rag

__all__ = [
    "FitzRagEngine",
    "run_fitz_rag",
    "create_fitz_rag_engine",
    "run",
]
