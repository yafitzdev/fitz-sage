# fitz_ai/engines/fitz_rag/retrieval/registry.py
"""
Retrieval Plugin Registry (YAML-based).

Discovers and loads retrieval plugins from YAML files in the plugins directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_rag.retrieval.loader import (
    RetrievalDependencies,
    RetrievalPipelineFromYaml,
    create_retrieval_pipeline,
    list_available_plugins,
)

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_rag.retrieval.steps import (
        ChatClient,
        Embedder,
        KeywordMatcherClient,
        Reranker,
        VectorClient,
    )


class PluginNotFoundError(Exception):
    """Raised when requested plugin doesn't exist."""

    pass


def get_retrieval_plugin(
    plugin_name: str,
    vector_client: "VectorClient",
    embedder: "Embedder",
    collection: str,
    reranker: "Reranker | None" = None,
    chat: "ChatClient | None" = None,
    keyword_matcher: "KeywordMatcherClient | None" = None,
    table_store: Any | None = None,
    top_k: int = 5,
    fetch_artifacts: bool = False,
) -> RetrievalPipelineFromYaml:
    """
    Get a retrieval plugin by name.

    Args:
        plugin_name: Name of the plugin (e.g., "dense", "dense_rerank")
        vector_client: Vector database client
        embedder: Embedding service
        collection: Collection name
        reranker: Optional reranking service
        chat: Optional fast-tier chat client for multi-query expansion
        keyword_matcher: Optional keyword matcher for exact term filtering
        table_store: Optional TableStore for CSV file queries
        top_k: Final number of chunks to return
        fetch_artifacts: Whether to fetch artifacts (always with score=1.0)

    Returns:
        Configured retrieval pipeline

    Raises:
        PluginNotFoundError: If plugin doesn't exist
    """
    try:
        return create_retrieval_pipeline(
            plugin_name=plugin_name,
            vector_client=vector_client,
            embedder=embedder,
            collection=collection,
            reranker=reranker,
            chat=chat,
            keyword_matcher=keyword_matcher,
            table_store=table_store,
            top_k=top_k,
            fetch_artifacts=fetch_artifacts,
        )
    except FileNotFoundError as e:
        raise PluginNotFoundError(str(e)) from e


def available_retrieval_plugins() -> list[str]:
    """List available retrieval plugins."""
    return list_available_plugins()


__all__ = [
    "get_retrieval_plugin",
    "available_retrieval_plugins",
    "PluginNotFoundError",
    "RetrievalPipelineFromYaml",
    "RetrievalDependencies",
]
