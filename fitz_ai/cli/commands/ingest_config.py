# fitz_ai/cli/commands/ingest_config.py
"""
Config building helpers for ingest command.

Functions for building typed config objects from raw config dicts.
"""

from __future__ import annotations


def build_chunking_router_config(ctx):
    """
    Build ChunkingRouterConfig from CLIContext.

    Uses ctx.chunk_size and ctx.chunk_overlap from typed config.

    Args:
        ctx: CLIContext with chunk_size and chunk_overlap attributes

    Returns:
        ChunkingRouterConfig with recursive chunker using ctx values.
    """
    from fitz_ai.engines.fitz_rag.config import (
        ChunkingRouterConfig,
        ExtensionChunkerConfig,
    )

    # Build default config using recursive chunker
    default = ExtensionChunkerConfig(
        plugin_name="recursive",
        kwargs={"chunk_size": ctx.chunk_size, "chunk_overlap": ctx.chunk_overlap},
    )

    return ChunkingRouterConfig(
        default=default,
        by_extension={},
        warn_on_fallback=False,
    )
