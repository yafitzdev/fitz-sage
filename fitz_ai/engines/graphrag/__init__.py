# fitz_ai/engines/graphrag/__init__.py
"""
GraphRAG Engine - Knowledge Graph-based Retrieval-Augmented Generation.

This engine implements Microsoft's GraphRAG paradigm which extracts
entities and relationships from documents, builds a knowledge graph
with community structure, and uses local/global search for retrieval.

Key features:
- Entity and relationship extraction using LLM
- Knowledge graph construction with NetworkX
- Community detection (Louvain/Leiden algorithms)
- Hierarchical community summarization
- Local search (entity-focused) for specific questions
- Global search (community-based) for broad questions
- Hybrid search combining both approaches

Public API:
    - GraphRAGEngine: Main engine class implementing KnowledgeEngine protocol
    - run_graphrag: Convenience function for quick queries
    - create_graphrag_engine: Factory for creating engine instances
    - GraphRAGConfig: Configuration dataclass

Search Modes:
    - "local": Entity-centric search for specific questions
    - "global": Community-based search for thematic questions
    - "hybrid": Combines both for comprehensive retrieval

Examples:
    Quick query:
    >>> from fitz_ai.engines.graphrag import run_graphrag
    >>>
    >>> documents = [
    ...     "Apple Inc. was founded by Steve Jobs and Steve Wozniak...",
    ...     "Microsoft was founded by Bill Gates and Paul Allen...",
    ... ]
    >>> answer = run_graphrag("Who founded Apple?", documents=documents)
    >>> print(answer.text)

    Reusable engine:
    >>> from fitz_ai.engines.graphrag import create_graphrag_engine, GraphRAGConfig
    >>>
    >>> config = GraphRAGConfig()
    >>> engine = create_graphrag_engine(config=config)
    >>>
    >>> # Add documents and build graph
    >>> engine.add_documents(my_documents)
    >>> engine.build_graph()
    >>> engine.build_communities()
    >>>
    >>> # Query multiple times
    >>> answer1 = engine.answer(Query(text="Question 1?"))
    >>> answer2 = engine.answer(Query(text="Question 2?"))

    Via universal runtime:
    >>> from fitz import run
    >>> answer = run("Who founded Apple?", engine="graphrag")

References:
    - Paper: https://arxiv.org/abs/2404.16130
    - GitHub: https://github.com/microsoft/graphrag
"""

# =============================================================================
# CONFIGURATION
# =============================================================================
from fitz_ai.engines.graphrag.config.schema import (
    GraphCommunityConfig,
    GraphExtractionConfig,
    GraphRAGConfig,
    GraphSearchConfig,
    GraphStorageConfig,
    load_graphrag_config,
)

# =============================================================================
# GRAPH COMPONENTS
# =============================================================================
from fitz_ai.engines.graphrag.graph.storage import (
    Community,
    Entity,
    KnowledgeGraph,
    Relationship,
)

# =============================================================================
# ENGINE
# =============================================================================
from fitz_ai.engines.graphrag.engine import GraphRAGEngine

# =============================================================================
# RUNTIME
# =============================================================================
from fitz_ai.engines.graphrag.runtime import (
    clear_engine_cache,
    create_graphrag_engine,
    run_graphrag,
)

__all__ = [
    # Engine
    "GraphRAGEngine",
    # Runtime
    "run_graphrag",
    "create_graphrag_engine",
    "clear_engine_cache",
    # Config
    "GraphRAGConfig",
    "GraphExtractionConfig",
    "GraphCommunityConfig",
    "GraphSearchConfig",
    "GraphStorageConfig",
    "load_graphrag_config",
    # Graph types
    "KnowledgeGraph",
    "Entity",
    "Relationship",
    "Community",
]


# =============================================================================
# ENGINE REGISTRATION
# =============================================================================


def _register_graphrag_engine():
    """Register GraphRAG with the engine registry."""
    from fitz_ai.engines.graphrag.config.schema import get_default_config_path
    from fitz_ai.runtime import EngineCapabilities, EngineRegistry

    def _create_graphrag_engine_factory(config) -> GraphRAGEngine:
        """Factory function for creating GraphRAG engine instances."""
        if config is None:
            config = GraphRAGConfig()
        elif isinstance(config, dict):
            config = GraphRAGConfig(
                extraction=GraphExtractionConfig(**config.get("extraction", {})),
                community=GraphCommunityConfig(**config.get("community", {})),
                search=GraphSearchConfig(**config.get("search", {})),
                storage=GraphStorageConfig(**config.get("storage", {})),
                llm_provider=config.get("llm_provider"),
                embedding_provider=config.get("embedding_provider"),
            )

        return GraphRAGEngine(config)

    def _graphrag_config_loader(config_path):
        """Load config for graphrag engine."""
        if config_path:
            return load_graphrag_config(config_path)
        return GraphRAGConfig()

    # Define capabilities
    capabilities = EngineCapabilities(
        supports_collections=False,  # Uses in-memory graph
        requires_documents_at_query=True,  # Must add docs before query
        supports_chat=False,  # No multi-turn yet
        supports_streaming=False,
        requires_config=False,  # Works with defaults
        requires_api_key=False,  # Uses configured LLM/embedding providers
        cli_query_message=(
            "GraphRAG requires documents to be loaded and graph to be built first.\n"
            "Use 'fitz quickstart <folder> \"question\" --engine graphrag' for one-off queries."
        ),
    )

    # Register with global registry
    registry = EngineRegistry.get_global()

    if "graphrag" not in registry.list():
        registry.register(
            name="graphrag",
            factory=_create_graphrag_engine_factory,
            description="GraphRAG: Knowledge graph-based RAG with entity extraction and community detection",
            config_type=GraphRAGConfig,
            config_loader=_graphrag_config_loader,
            default_config_path=get_default_config_path,
            capabilities=capabilities,
        )


# Perform registration when module is imported
_register_graphrag_engine()
