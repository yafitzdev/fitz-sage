# fitz_ai/engines/fitz_krag/config/schema.py
"""
Configuration schema for Fitz KRAG (Knowledge Routing Augmented Generation) engine.

KRAG uses knowledge-type-aware access strategies instead of uniform chunk-based
retrieval. It stores raw files and symbol indexes, retrieves by address (pointer
to code symbol / document section), then reads content on demand.
"""

from __future__ import annotations

from pydantic import Field

from fitz_ai.engines.fitz_rag.config.schema import BasePluginConfig, PluginKwargs


class FitzKragConfig(BasePluginConfig):
    """
    Fitz KRAG configuration.

    Minimal config:
    ```yaml
    chat: cohere
    embedding: cohere
    collection: my_project
    ```
    """

    # ==========================================================================
    # Core Plugins (shared infrastructure)
    # ==========================================================================

    chat: str = Field(
        default="cohere",
        description="Chat plugin: 'provider' or 'provider/model'",
    )

    embedding: str = Field(
        default="cohere",
        description="Embedding plugin: 'provider' or 'provider/model'",
    )

    vector_db: str = Field(
        default="pgvector",
        description="Vector DB plugin (pgvector only)",
    )

    rerank: str | None = Field(
        default=None,
        description="Reranker plugin. None = disabled.",
    )

    # ==========================================================================
    # Collection
    # ==========================================================================

    collection: str = Field(
        ...,
        description="Collection name (required)",
    )

    # ==========================================================================
    # Code Strategy
    # ==========================================================================

    code_languages: list[str] = Field(
        default=["python", "typescript", "java", "go"],
        description="Enabled code languages for ingestion",
    )

    summary_batch_size: int = Field(
        default=15,
        ge=1,
        description="Number of symbols per LLM summarization batch",
    )

    max_expansion_depth: int = Field(
        default=1,
        ge=0,
        description="Max depth for code context expansion (imports, class context)",
    )

    include_class_context: bool = Field(
        default=True,
        description="Include class signature + __init__ when expanding methods",
    )

    # ==========================================================================
    # Retrieval
    # ==========================================================================

    top_addresses: int = Field(
        default=10,
        ge=1,
        description="Number of addresses to retrieve before reading",
    )

    top_read: int = Field(
        default=5,
        ge=1,
        description="Number of top addresses to read content for",
    )

    keyword_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for keyword search in hybrid merge",
    )

    semantic_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search in hybrid merge",
    )

    fallback_to_chunks: bool = Field(
        default=True,
        description="Fall back to chunk-based search when code search returns few results",
    )

    section_bm25_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 search in section hybrid merge",
    )

    section_semantic_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search in section hybrid merge",
    )

    # ==========================================================================
    # Context Assembly
    # ==========================================================================

    max_context_tokens: int = Field(
        default=8000,
        ge=100,
        description="Max tokens in assembled context for LLM",
    )

    include_file_header: bool = Field(
        default=True,
        description="Include file path header in context blocks",
    )

    # ==========================================================================
    # Generation
    # ==========================================================================

    enable_citations: bool = Field(
        default=True,
        description="Enable [S1], [S2] citation markers in answers",
    )

    strict_grounding: bool = Field(
        default=True,
        description="Only generate answers from provided context",
    )

    # ==========================================================================
    # Plugin kwargs
    # ==========================================================================

    chat_kwargs: PluginKwargs = Field(
        default_factory=PluginKwargs,
        description="Additional kwargs for chat plugin",
    )

    embedding_kwargs: PluginKwargs = Field(
        default_factory=PluginKwargs,
        description="Additional kwargs for embedding plugin",
    )

    vector_db_kwargs: PluginKwargs = Field(
        default_factory=PluginKwargs,
        description="pgvector config: mode, connection_string, hnsw_m, etc.",
    )

    # ==========================================================================
    # Logging
    # ==========================================================================

    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR",
    )
