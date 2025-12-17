# fitz/ingest/ingestion/registry.py
"""
Ingestion plugin registry.

This is a thin wrapper around fitz.core.registry.
All the actual logic lives there - this file just provides
backwards-compatible imports.

For new code, prefer importing directly from fitz.core.registry:
    from fitz.core.registry import get_ingest_plugin, available_ingest_plugins
"""

from fitz.core.registry import (
    # Main functions
    get_ingest_plugin,
    available_ingest_plugins,
    # Registry (if needed for advanced use)
    INGEST_REGISTRY,
    # Errors
    PluginRegistryError,
    PluginNotFoundError,
    DuplicatePluginError,
)

# Backwards compatibility - old name was REGISTRY
REGISTRY = INGEST_REGISTRY._registry  # Direct access for legacy code

__all__ = [
    "get_ingest_plugin",
    "available_ingest_plugins",
    "INGEST_REGISTRY",
    "REGISTRY",
    "PluginRegistryError",
    "PluginNotFoundError",
    "DuplicatePluginError",
]