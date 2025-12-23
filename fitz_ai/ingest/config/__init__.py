# fitz_ai/ingest/config/__init__.py
"""
Configuration schemas for the ingestion pipeline.

Exports:
- ChunkingRouterConfig: Router config with per-extension chunking
- ExtensionChunkerConfig: Per-extension chunker config
- IngestConfig: Top-level ingestion config
- IngesterConfig: Ingestion plugin config
"""

from fitz_ai.ingest.config.schema import (
    ChunkingRouterConfig,
    ExtensionChunkerConfig,
    IngestConfig,
    IngesterConfig,
)

__all__ = [
    "ChunkingRouterConfig",
    "ExtensionChunkerConfig",
    "IngestConfig",
    "IngesterConfig",
]