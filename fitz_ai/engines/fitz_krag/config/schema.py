# fitz_ai/engines/fitz_krag/config/schema.py
"""
Configuration schema for Fitz KRAG (Knowledge Routing Augmented Generation) engine.

KRAG uses knowledge-type-aware access strategies instead of uniform chunk-based
retrieval. It stores raw files and symbol indexes, retrieves by address (pointer
to code symbol / document section), then reads content on demand.
"""

from __future__ import annotations

from pydantic import Field

from fitz_ai.core.config import BasePluginConfig, PluginKwargs


class FitzKragConfig(BasePluginConfig):
    """
    Fitz KRAG configuration.

    Minimal config:
    ```yaml
    chat_fast: ollama/qwen3.5:0.6b
    chat_balanced: ollama/qwen2.5:7b
    chat_smart: ollama/qwen2.5:14b
    embedding: ollama/nomic-embed-text
    collection: my_project
    ```
    """

    # ==========================================================================
    # Core Plugins (shared infrastructure)
    # ==========================================================================

    chat_fast: str = Field(
        default="ollama/qwen3.5:0.6b",
        description="Chat model for detection, guardrails (provider/model)",
    )

    chat_balanced: str = Field(
        default="ollama/qwen2.5:7b",
        description="Chat model for general queries (provider/model)",
    )

    chat_smart: str = Field(
        default="ollama/qwen2.5:14b",
        description="Chat model for complex generation (provider/model)",
    )

    embedding: str = Field(
        default="ollama/nomic-embed-text",
        description="Embedding model (provider/model)",
    )

    vector_db: str = Field(
        default="pgvector",
        description="Vector DB plugin (pgvector only)",
    )

    rerank: str | None = Field(
        default=None,
        description="Reranker plugin. None = disabled.",
    )

    vision: str | None = Field(
        default=None,
        description="Vision/VLM plugin for image description. None = disabled.",
    )

    parser: str = Field(
        default="docling",
        description="Document parser: 'docling', 'docling_vision', or 'glm_ocr' (hybrid pypdfium2 + GLM-OCR via ollama)",
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

    code_search_mode: str = Field(
        default="auto",
        description=(
            "Code search mode: 'auto' = LLM structural search when chat "
            "available with hybrid fallback, 'hybrid' = BM25 + semantic only"
        ),
    )

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

    max_reference_expansions: int = Field(
        default=3,
        ge=0,
        description="Max same-file referenced symbols to include as context (0 = disabled)",
    )

    include_import_summaries: bool = Field(
        default=True,
        description="Include summaries of imported symbols as context",
    )

    max_import_expansions: int = Field(
        default=5,
        ge=0,
        description="Max imported symbol summaries to include as context",
    )

    # ==========================================================================
    # Retrieval
    # ==========================================================================

    top_addresses: int = Field(
        default=50,
        ge=1,
        description="Number of addresses to retrieve before reading",
    )

    top_read: int = Field(
        default=50,
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

    include_section_context: bool = Field(
        default=True,
        description="Include parent breadcrumb and child TOC for section addresses",
    )

    # ==========================================================================
    # Table Strategy
    # ==========================================================================

    table_extensions: list[str] = Field(
        default=[".csv", ".tsv"],
        description="File extensions to ingest as tables",
    )

    table_keyword_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for keyword search in table hybrid merge",
    )

    table_semantic_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search in table hybrid merge",
    )

    max_table_results: int = Field(
        default=100,
        ge=1,
        description="Max SQL result rows to include in context",
    )

    # ==========================================================================
    # Context Assembly
    # ==========================================================================

    max_context_tokens: int = Field(
        default=48000,
        ge=100,
        description="Max tokens in assembled context for LLM",
    )

    include_file_header: bool = Field(
        default=True,
        description="Include file path header in context blocks",
    )

    # ==========================================================================
    # Guardrails
    # ==========================================================================

    enable_guardrails: bool = Field(
        default=True,
        description="Enable epistemic guardrails (constraint checking before generation)",
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
    # Cloud
    # ==========================================================================

    cloud: dict = Field(
        default_factory=dict,
        description="Fitz Cloud config (enabled, api_key, org_key, etc.)",
    )

    # ==========================================================================
    # Detection
    # ==========================================================================

    enable_detection: bool = Field(
        default=True,
        description="Enable shared detection (temporal, comparison, expansion awareness)",
    )

    # ==========================================================================
    # Query Intelligence
    # ==========================================================================

    enable_query_rewriting: bool = Field(
        default=True,
        description="Enable LLM-based query rewriting for retrieval optimization",
    )

    enable_hyde: bool = Field(
        default=True,
        description="Enable HyDE (Hypothetical Document Embeddings) for improved recall",
    )

    enable_multi_query: bool = Field(
        default=True,
        description="Enable multi-query expansion for long/complex queries",
    )

    multi_query_min_length: int = Field(
        default=300,
        ge=50,
        description="Minimum query character length to trigger multi-query expansion",
    )

    # ==========================================================================
    # Reranking
    # ==========================================================================

    rerank_k: int = Field(
        default=10,
        ge=1,
        description="Number of addresses to keep after reranking",
    )

    rerank_min_addresses: int = Field(
        default=20,
        ge=1,
        description="Minimum addresses before reranking is applied (skip if fewer)",
    )

    # ==========================================================================
    # BM25 Code Search
    # ==========================================================================

    code_bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 search in code hybrid merge",
    )

    # ==========================================================================
    # Enrichment
    # ==========================================================================

    enable_enrichment: bool = Field(
        default=True,
        description="Enable keyword/entity extraction during ingestion",
    )

    # ==========================================================================
    # Multi-Hop
    # ==========================================================================

    enable_multi_hop: bool = Field(
        default=False,
        description="Enable multi-hop iterative retrieval for complex queries",
    )

    max_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum retrieval hops for multi-hop reasoning",
    )

    # ==========================================================================
    # Hierarchy
    # ==========================================================================

    enable_hierarchy: bool = Field(
        default=True,
        description="Enable L1/L2 hierarchical summaries during ingestion",
    )

    # ==========================================================================
    # Plugin kwargs
    # ==========================================================================

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
