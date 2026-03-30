# fitz_sage/ingestion/chunking/config.py
"""Configuration schemas for the chunking subsystem."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
