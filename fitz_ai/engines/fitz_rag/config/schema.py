# fitz_ai/engines/fitz_rag/config/schema.py
"""
Configuration schema for Fitz RAG engine.

Philosophy:
- Convention over configuration: sensible defaults for 80% use cases
- Plugin strings: "provider" or "provider/model" format
- None = disabled: No enabled flags, just set to None
- Flat structure: Minimal nesting

Example minimal config (3 lines):
```yaml
chat: anthropic/claude-sonnet-4
embedding: openai/text-embedding-3-small
collection: my_docs
```

Example with all features:
```yaml
chat: cohere/command-r-plus
embedding: cohere/embed-english-v3.0
vector_db: qdrant
rerank: cohere/rerank-english-v3.0
vision: openai/gpt-4o
collection: my_docs
top_k: 10
enable_citations: true
strict_grounding: true
```
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
    from fitz_ai.cloud.config import CloudConfig
except ImportError:
    CloudConfig = None  # Cloud is optional


# =============================================================================
# Base Config with Common Settings
# =============================================================================


class BasePluginConfig(BaseModel):
    """Base class to avoid repeating model_config."""

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Plugin Configuration
# =============================================================================


class PluginKwargs(BaseModel):
    """
    Additional kwargs for plugin initialization.

    Common parameters used by LLM plugins. Plugins ignore fields they don't need.
    Allows extra fields for plugin-specific configuration.
    """

    model: str | None = Field(
        default=None,
        description="Model override (e.g., 'gpt-4', 'claude-sonnet-4')",
    )

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (0.0-2.0)",
    )

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate",
    )

    timeout: int | None = Field(
        default=None,
        ge=1,
        description="Request timeout in seconds",
    )

    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter",
    )

    host: str | None = Field(
        default=None,
        description="Host for self-hosted services (e.g., Qdrant, Ollama)",
    )

    port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Port for self-hosted services",
    )

    api_key: str | None = Field(
        default=None,
        description="API key override (use environment variables instead)",
    )

    # Allow plugins to add their own fields
    model_config = ConfigDict(extra="allow")


# =============================================================================
# Simplified Main Configuration
# =============================================================================


class FitzRagConfig(BasePluginConfig):
    """
    Fitz RAG configuration with sensible defaults.

    Philosophy:
    - Strings for plugins: "cohere" or "cohere/command-r-plus"
    - None = disabled: No enabled flags
    - Flat structure: Minimal nesting
    - Defaults for common cases: Works out of the box

    Minimal config (3 lines):
    ```yaml
    chat: anthropic/claude-sonnet-4
    embedding: openai/text-embedding-3-small
    collection: my_docs
    ```

    Full config example:
    ```yaml
    # Core plugins (required)
    chat: cohere/command-r-plus
    embedding: cohere/embed-english-v3.0
    vector_db: pgvector

    # Optional features (None = disabled)
    rerank: cohere/rerank-english-v3.0
    vision: null

    # Retrieval
    retrieval_plugin: dense
    collection: my_docs
    top_k: 5

    # Generation
    enable_citations: true
    strict_grounding: true
    max_chunks: 8

    # Chunking
    chunk_size: 512
    parser: docling

    # Cloud (optional)
    cloud: null
    ```
    """

    # ==========================================================================
    # Core Plugins (Required)
    # ==========================================================================

    chat: str = Field(
        default="cohere",
        description="Chat plugin: 'provider' or 'provider/model' (e.g., 'anthropic/claude-sonnet-4')",
    )

    embedding: str = Field(
        default="cohere",
        description="Embedding plugin: 'provider' or 'provider/model' (e.g., 'openai/text-embedding-3-small')",
    )

    vector_db: str = Field(
        default="pgvector",
        description="Vector DB plugin: 'pgvector' (default), 'qdrant', 'pinecone', etc.",
    )

    # ==========================================================================
    # Optional Features (None = Disabled)
    # ==========================================================================

    rerank: str | None = Field(
        default=None,
        description="Reranker plugin: 'provider' or 'provider/model'. None = disabled.",
    )

    vision: str | None = Field(
        default=None,
        description="Vision/VLM plugin for image parsing: 'provider' or 'provider/model'. None = disabled.",
    )

    # ==========================================================================
    # Retrieval Configuration
    # ==========================================================================

    retrieval_plugin: str = Field(
        default="dense",
        description="Retrieval pipeline plugin: 'dense', 'dense_rerank', etc.",
    )

    collection: str = Field(
        ...,
        description="Vector DB collection name (required)",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        description="Number of chunks to retrieve",
    )

    fetch_artifacts: bool = Field(
        default=False,
        description="Fetch project artifacts (navigation index, etc.) with every query",
    )

    # ==========================================================================
    # Generation Configuration (Flattened from RGSConfig)
    # ==========================================================================

    enable_citations: bool = Field(
        default=True,
        description="Enable citation markers in answers ([S1], [S2], etc.)",
    )

    strict_grounding: bool = Field(
        default=True,
        description="Only generate answers based on provided context",
    )

    max_chunks: int = Field(
        default=8,
        ge=1,
        description="Maximum chunks to include in generation context",
    )

    include_query_in_context: bool = Field(
        default=True,
        description="Include the query in the generation prompt",
    )

    max_answer_chars: int | None = Field(
        default=None,
        description="Maximum answer length in characters. None = no limit.",
    )

    # ==========================================================================
    # Chunking Configuration (Simplified)
    # ==========================================================================

    chunk_size: int = Field(
        default=512,
        ge=50,
        description="Default chunk size in tokens",
    )

    chunk_overlap: int = Field(
        default=0,
        ge=0,
        description="Chunk overlap in tokens",
    )

    parser: str = Field(
        default="docling",
        description="Parser plugin: 'docling' (no VLM) or 'docling_vision' (with VLM)",
    )

    # ==========================================================================
    # Cloud Configuration (Optional)
    # ==========================================================================

    cloud: CloudConfig | None = Field(
        default=None,
        description="Cloud cache configuration. None = disabled.",
    )

    # ==========================================================================
    # Logging
    # ==========================================================================

    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR",
    )

    # ==========================================================================
    # Advanced: Plugin kwargs override
    # ==========================================================================

    chat_kwargs: PluginKwargs = Field(
        default_factory=PluginKwargs,
        description="Additional kwargs for chat plugin (model, temperature, max_tokens, etc.)",
    )

    embedding_kwargs: PluginKwargs = Field(
        default_factory=PluginKwargs,
        description="Additional kwargs for embedding plugin (model, etc.)",
    )

    vector_db_kwargs: PluginKwargs = Field(
        default_factory=PluginKwargs,
        description="Additional kwargs for vector DB plugin (host, port, etc.)",
    )

    rerank_kwargs: PluginKwargs = Field(
        default_factory=PluginKwargs,
        description="Additional kwargs for reranker plugin (model, etc.)",
    )

    vision_kwargs: PluginKwargs = Field(
        default_factory=PluginKwargs,
        description="Additional kwargs for vision plugin (model, temperature, max_tokens, etc.)",
    )


# =============================================================================
# Chunking Router Configuration (Helper Classes)
# =============================================================================


class ExtensionChunkerConfig(BaseModel):
    """Configuration for a chunker plugin (used by ChunkingRouter)."""

    plugin_name: str = Field(
        description="Chunker plugin name (e.g., 'recursive', 'simple')",
    )

    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration",
    )

    model_config = ConfigDict(extra="forbid")


class ChunkingRouterConfig(BaseModel):
    """Configuration for the ChunkingRouter."""

    default: ExtensionChunkerConfig = Field(
        description="Default chunker for all extensions",
    )

    by_extension: dict[str, ExtensionChunkerConfig] = Field(
        default_factory=dict,
        description="Extension-specific chunker overrides",
    )

    warn_on_fallback: bool = Field(
        default=False,
        description="Warn when using default chunker",
    )

    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context):
        """Normalize extension keys to lowercase with dot prefix."""
        normalized = {}
        for ext, config in self.by_extension.items():
            ext_lower = ext.lower()
            normalized_ext = ext_lower if ext_lower.startswith(".") else f".{ext_lower}"
            normalized[normalized_ext] = config
        self.by_extension = normalized
