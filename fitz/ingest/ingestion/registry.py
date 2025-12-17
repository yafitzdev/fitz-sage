# fitz/ingest/ingestion/registry.py
"""
Ingestion plugin registry.

Thin wrapper around fitz.core.registry.
"""

from fitz.core.registry import (
    get_ingest_plugin,
    available_ingest_plugins,
    INGEST_REGISTRY,
    PluginRegistryError,
    PluginNotFoundError,
    DuplicatePluginError,
)

__all__ = [
    "get_ingest_plugin",
    "available_ingest_plugins",
    "INGEST_REGISTRY",
    "PluginRegistryError",
    "PluginNotFoundError",
    "DuplicatePluginError",
]