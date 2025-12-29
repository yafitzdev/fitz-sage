# fitz_ai/engines/graphrag/runtime.py
"""
GraphRAG Runtime - Convenience functions and engine management.

Provides simple interfaces for using GraphRAG without managing
engine instances directly.
"""

from typing import Dict, List, Optional

from fitz_ai.core import Answer, Query
from fitz_ai.engines.graphrag.config.schema import GraphRAGConfig
from fitz_ai.engines.graphrag.engine import GraphRAGEngine

# Engine cache for reuse
_engine_cache: Dict[str, GraphRAGEngine] = {}


def create_graphrag_engine(
    config: Optional[GraphRAGConfig] = None,
    cache_key: Optional[str] = None,
) -> GraphRAGEngine:
    """
    Create or retrieve a cached GraphRAG engine.

    Args:
        config: Engine configuration. Uses defaults if None.
        cache_key: Optional key for caching. If provided and engine
                   exists in cache, returns cached instance.

    Returns:
        GraphRAGEngine instance
    """
    if cache_key and cache_key in _engine_cache:
        return _engine_cache[cache_key]

    engine = GraphRAGEngine(config)

    if cache_key:
        _engine_cache[cache_key] = engine

    return engine


def run_graphrag(
    question: str,
    documents: Optional[List[str]] = None,
    config: Optional[GraphRAGConfig] = None,
    search_mode: str = "local",
) -> Answer:
    """
    Run a GraphRAG query with documents.

    This is a convenience function for one-off queries. For multiple
    queries on the same documents, use create_graphrag_engine() to
    avoid rebuilding the graph.

    Args:
        question: Question to answer
        documents: List of document texts (required for first query)
        config: Engine configuration
        search_mode: Search mode: "local", "global", or "hybrid"

    Returns:
        Answer object

    Examples:
        >>> docs = ["Document 1...", "Document 2..."]
        >>> answer = run_graphrag("What is X?", documents=docs)
        >>> print(answer.text)
    """
    # Create engine (not cached since documents may differ)
    engine = GraphRAGEngine(config)

    if documents:
        engine.add_documents(documents)

    # Override search mode in config
    engine.config.search.default_mode = search_mode

    return engine.answer(Query(text=question))


def clear_engine_cache() -> None:
    """Clear all cached engine instances."""
    _engine_cache.clear()
