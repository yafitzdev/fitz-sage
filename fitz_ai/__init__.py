# fitz_ai/__init__.py
"""
Fitz - Local-First RAG Framework & Engine Platform

Fitz is a paradigm-agnostic knowledge engine platform that supports multiple
approaches to knowledge retrieval and synthesis.

Quick Start:
    >>> from fitz import run
    >>> answer = run("What is quantum computing?")
    >>> print(answer.text)

Public API:
    Core Types:
        - Query: Input to engines
        - Answer: Output from engines
        - Provenance: Source attribution
        - Constraints: Query-time constraints

    Runtime:
        - run: Universal entry point (any engine)
        - create_engine: Factory for creating engines
        - list_engines: List available engines

Architecture:
    fitz_ai/
    ├── core/              # Paradigm-agnostic contracts
    ├── engines/           # Engine implementations
    │   └── fitz_krag/     # Knowledge Routing Augmented Generation
    ├── runtime/           # Multi-engine orchestration
    ├── llm/               # LLM service (chat, embedding, rerank)
    ├── vector_db/         # Vector database service
    └── ingest/            # Document ingestion

Philosophy:
    Knowledge → Engine → Answer

    Engines are black boxes that transform queries into answers.
    The platform only cares about the interface, not the implementation.

Examples:
    Simple query:
    >>> from fitz import run
    >>> answer = run("What is quantum computing?")

    With constraints:
    >>> from fitz import run, Constraints
    >>> constraints = Constraints(max_sources=5)
    >>> answer = run("Explain entanglement", constraints=constraints)

    Specific engine:
    >>> answer = run("What is X?", engine="fitz_krag")

    Reusable engine:
    >>> from fitz import create_engine, Query
    >>> engine = create_engine("fitz_krag")
    >>> query = Query(text="What is Y?")
    >>> answer = engine.answer(query)
"""

__version__ = "0.10.0"

# =============================================================================
# LAZY IMPORTS
# =============================================================================
# Heavy modules (engines, runtime) are only imported when accessed.
# This keeps CLI startup fast.


def __getattr__(name: str):
    """Lazy import for heavy modules."""
    # Core types (lightweight, always available)
    if name in (
        "Answer",
        "ConfigurationError",
        "Constraints",
        "EngineError",
        "GenerationError",
        "KnowledgeEngine",
        "KnowledgeError",
        "Provenance",
        "Query",
        "QueryError",
        "TimeoutError",
        "UnsupportedOperationError",
    ):
        from fitz_ai import core

        return getattr(core, name)

    # Runtime (heavy - discovers all engines)
    if name in (
        "create_engine",
        "get_engine_registry",
        "list_engines",
        "list_engines_with_info",
        "run",
    ):
        from fitz_ai import runtime

        return getattr(runtime, name)

    # SDK
    if name == "fitz":
        from fitz_ai import sdk

        return getattr(sdk, name)

    raise AttributeError(f"module 'fitz_ai' has no attribute {name!r}")


# =============================================================================
# MODULE-LEVEL SDK (matches CLI: fitz point, fitz query)
# =============================================================================

_default_fitz = None


def _get_default_fitz():
    """Get or create the default fitz instance."""
    global _default_fitz
    if _default_fitz is None:
        from fitz_ai.sdk import fitz

        _default_fitz = fitz()
    return _default_fitz


def point(source, collection: str = None):
    """
    Point at a folder for immediate querying with background indexing.

    Module-level convenience function matching `fitz point` CLI.

    Args:
        source: Path to file or directory.
        collection: Collection name (uses default if not specified).

    Examples:
        >>> import fitz_ai
        >>> fitz_ai.point("./docs")
        >>> answer = fitz_ai.query("What is X?")
    """
    global _default_fitz
    if collection is not None:
        from fitz_ai.sdk import fitz

        _default_fitz = fitz(collection=collection)
    f = _get_default_fitz()
    return f.point(source)


def query(question: str, top_k: int = None):
    """
    Query the knowledge base.

    Module-level convenience function matching `fitz query` CLI.

    Args:
        question: The question to ask.
        top_k: Number of chunks to retrieve (uses config default if not specified).

    Returns:
        Answer with text and provenance.

    Examples:
        >>> import fitz_ai
        >>> fitz_ai.point("./docs")
        >>> answer = fitz_ai.query("What is the refund policy?")
        >>> print(answer.text)
    """
    f = _get_default_fitz()
    return f.ask(question, top_k=top_k)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Core Protocol
    "KnowledgeEngine",
    # Core Types
    "Query",
    "Answer",
    "Provenance",
    "Constraints",
    # Core Exceptions
    "EngineError",
    "QueryError",
    "KnowledgeError",
    "GenerationError",
    "ConfigurationError",
    "TimeoutError",
    "UnsupportedOperationError",
    # Universal Runtime
    "run",
    "create_engine",
    "list_engines",
    "list_engines_with_info",
    "get_engine_registry",
    # SDK
    "fitz",
    "point",
    "query",
]
