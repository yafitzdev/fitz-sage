# fitz_ai/ingest/config/schema.py
"""
Configuration schemas for the ingestion pipeline.

This module defines Pydantic models for:
- IngesterConfig: Ingestion plugin configuration
- ExtensionChunkerConfig: Per-extension chunker configuration
- ChunkingRouterConfig: Router configuration with per-extension mapping
- IngestConfig: Top-level ingestion configuration
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IngesterConfig(BaseModel):
    """Configuration for the ingestion plugin."""

    plugin_name: str = Field(..., description="Ingester plugin name")
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Ingester init kwargs"
    )

    model_config = ConfigDict(extra="forbid")


class ExtensionChunkerConfig(BaseModel):
    """
    Configuration for a chunker assigned to a specific file extension.

    Example:
        >>> ExtensionChunkerConfig(
        ...     plugin_name="markdown",
        ...     kwargs={"max_tokens": 800, "preserve_headers": True}
        ... )
    """

    plugin_name: str = Field(..., description="Chunker plugin name")
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Chunker-specific parameters"
    )

    model_config = ConfigDict(extra="forbid")


class ChunkingRouterConfig(BaseModel):
    """
    Configuration for the ChunkingRouter.

    Supports file-type specific chunking with a default fallback.

    Attributes:
        default: Default chunker for extensions not in by_extension.
        by_extension: Mapping of file extensions to chunker configs.
        warn_on_fallback: Whether to log warnings when using default chunker.

    Example YAML:
        chunking:
          default:
            plugin_name: simple
            kwargs:
              chunk_size: 1000
              chunk_overlap: 0

          by_extension:
            .md:
              plugin_name: markdown
              kwargs:
                max_tokens: 800
            .py:
              plugin_name: python_code
              kwargs:
                chunk_by: function
            .pdf:
              plugin_name: pdf_sections
              kwargs:
                max_section_chars: 2000

          warn_on_fallback: true
    """

    default: ExtensionChunkerConfig = Field(
        ..., description="Default chunker for unknown extensions"
    )
    by_extension: Dict[str, ExtensionChunkerConfig] = Field(
        default_factory=dict,
        description="Mapping of extensions to chunker configs",
    )
    warn_on_fallback: bool = Field(
        default=True,
        description="Log warning when using default chunker for unknown extension",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("by_extension", mode="before")
    @classmethod
    def normalize_extensions(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all extension keys start with a dot and are lowercase."""
        if not isinstance(v, dict):
            return v

        normalized = {}
        for ext, config in v.items():
            norm_ext = ext.lower()
            if not norm_ext.startswith("."):
                norm_ext = f".{norm_ext}"
            normalized[norm_ext] = config

        return normalized


class IngestConfig(BaseModel):
    """
    Central configuration for the entire ingestion pipeline.

    Example YAML:
        ingester:
          plugin_name: local
          kwargs: {}

        chunking:
          default:
            plugin_name: simple
            kwargs:
              chunk_size: 1000
          by_extension:
            .md:
              plugin_name: markdown
              kwargs:
                max_tokens: 800

        collection: my_docs
    """

    ingester: IngesterConfig = Field(..., description="Ingestion plugin config")
    chunking: ChunkingRouterConfig = Field(
        ..., description="Chunking router config"
    )
    collection: str = Field(..., description="Target vector DB collection")

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "IngesterConfig",
    "ExtensionChunkerConfig",
    "ChunkingRouterConfig",
    "IngestConfig",
]