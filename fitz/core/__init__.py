"""
Fitz Core - Paradigm-agnostic contracts for knowledge engines.

This module defines the stable abstractions that all engines must implement.
The core philosophy is: Knowledge → Engine → Answer.

Public API:
    - KnowledgeEngine: Protocol that all engines implement
    - Query: Input to engines
    - Answer: Output from engines
    - Provenance: Source attribution
    - Constraints: Query-time constraints
    - Exceptions: Standard error hierarchy

Examples:
    Using the core abstractions:
    >>> from fitz.core import Query, Constraints
    >>> from fitz.engines.classic_rag import ClassicRagEngine
    >>>
    >>> # Create an engine (engine-specific)
    >>> engine = ClassicRagEngine(config)
    >>>
    >>> # Use core abstractions (paradigm-agnostic)
    >>> query = Query(
    ...     text="What is quantum computing?",
    ...     constraints=Constraints(max_sources=5)
    ... )
    >>> answer = engine.answer(query)
    >>>
    >>> # Access results (paradigm-agnostic)
    >>> print(answer.text)
    >>> for source in answer.provenance:
    ...     print(f"Source: {source.source_id}")
"""

from .answer import Answer
from .constraints import Constraints

# Core protocol
from .engine import KnowledgeEngine

# Core exceptions
from .exceptions import (
    ConfigurationError,
    EngineError,
    GenerationError,
    KnowledgeError,
    QueryError,
    TimeoutError,
    UnsupportedOperationError,
)
from .provenance import Provenance

# Core types
from .query import Query

__all__ = [
    # Protocol
    "KnowledgeEngine",
    # Types
    "Query",
    "Answer",
    "Provenance",
    "Constraints",
    # Exceptions
    "EngineError",
    "QueryError",
    "KnowledgeError",
    "GenerationError",
    "ConfigurationError",
    "TimeoutError",
    "UnsupportedOperationError",
]
