# fitz_ai/engines/fitz_rag/retrieval/__init__.py
"""
Retrieval subsystem for Fitz RAG.

Provides YAML-based retrieval plugins with composable step pipelines.

Usage:
    from fitz_ai.engines.fitz_rag.retrieval import (
        get_retrieval_plugin,
        available_retrieval_plugins,
    )

    # Or import steps directly
    from fitz_ai.engines.fitz_rag.retrieval.steps import (
        VectorSearchStep,
        RerankStep,
        ThresholdStep,
        LimitStep,
        DedupeStep,
    )
"""

# Re-export registry functions
from .registry import (
    PluginNotFoundError,
    RetrievalDependencies,
    RetrievalPipelineFromYaml,
    available_retrieval_plugins,
    get_retrieval_plugin,
)

# Re-export step classes for convenience
from .steps import (
    STEP_REGISTRY,
    DedupeStep,
    LimitStep,
    RerankStep,
    RetrievalStep,
    ThresholdStep,
    VectorSearchStep,
    get_step_class,
    list_available_steps,
)

__all__ = [
    # Registry functions
    "get_retrieval_plugin",
    "available_retrieval_plugins",
    "PluginNotFoundError",
    "RetrievalPipelineFromYaml",
    "RetrievalDependencies",
    # Step classes
    "RetrievalStep",
    "VectorSearchStep",
    "RerankStep",
    "ThresholdStep",
    "LimitStep",
    "DedupeStep",
    # Registry
    "STEP_REGISTRY",
    "get_step_class",
    "list_available_steps",
]
