# fitz_ai/cli/commands/ingest_config.py
"""
Config building helpers for ingest command.

Functions for building typed config objects from raw config dicts.
"""

from __future__ import annotations


def build_chunking_router_config(config: dict):
    """
    Build ChunkingRouterConfig from fitz.yaml config.

    Expected config structure:
        chunking:
          default:
            plugin_name: simple
            kwargs:
              chunk_size: 1000
              chunk_overlap: 0
          by_extension:
            .md:
              plugin_name: markdown
              kwargs: {...}
          warn_on_fallback: true
    """
    from fitz_ai.engines.fitz_rag.config import (
        ChunkingRouterConfig,
        ExtensionChunkerConfig,
    )

    chunking = config.get("chunking", {})

    # Build default config
    default_cfg = chunking.get("default", {})
    default = ExtensionChunkerConfig(
        plugin_name=default_cfg.get("plugin_name", "simple"),
        kwargs=default_cfg.get("kwargs", {"chunk_size": 1000, "chunk_overlap": 0}),
    )

    # Build per-extension configs
    by_extension = {}
    for ext, ext_cfg in chunking.get("by_extension", {}).items():
        by_extension[ext] = ExtensionChunkerConfig(
            plugin_name=ext_cfg.get("plugin_name", "simple"),
            kwargs=ext_cfg.get("kwargs", {}),
        )

    return ChunkingRouterConfig(
        default=default,
        by_extension=by_extension,
        warn_on_fallback=chunking.get("warn_on_fallback", False),
    )
