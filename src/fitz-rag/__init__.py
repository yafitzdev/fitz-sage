# src/fitz_rag/__init__.py
"""
fitz-rag: A modular, plugin-based RAG framework for multi-source retrieval
and LLM context assembly.

This package provides:
- A plugin system for defining retrieval sources
- Deterministic and semantic retrieval strategies
- RAGContextBuilder for multi-source context construction
- Prompt builder utilities
- Retriever abstractions for Qdrant and embedding-based retrieval

Typical usage:

    from fitz_rag.sourcer.rag_base import (
        RAGContextBuilder,
        load_source_configs,
        SourceConfig,
        ArtefactRetrievalStrategy,
    )
"""

# Re-export core API elements for convenient import
from .sourcer.rag_base import (
    RAGContextBuilder,
    load_source_configs,
    SourceConfig,
    ArtefactRetrievalStrategy,
)

from .sourcer.prompt_builder import (
    build_user_prompt,
    build_rag_block,
)

__all__ = [
    "RAGContextBuilder",
    "load_source_configs",
    "SourceConfig",
    "ArtefactRetrievalStrategy",
    "build_user_prompt",
    "build_rag_block",
]
