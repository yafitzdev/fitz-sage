# fitz/ingest/chunking/registry.py
"""
Chunking plugin registry.

This is a thin wrapper around fitz.core.registry.
All the actual logic lives there - this file just provides
backwards-compatible imports.

For new code, prefer importing directly from fitz.core.registry:
    from fitz.core.registry import get_chunking_plugin, available_chunking_plugins
"""

from fitz.core.registry import (
    # Main functions
    get_chunking_plugin,
    available_chunking_plugins,
    # Registry (if needed for advanced use)
    CHUNKING_REGISTRY,
    # Errors
    PluginRegistryError,
    PluginNotFoundError,
    DuplicatePluginError,
)

__all__ = [
    "get_chunking_plugin",
    "available_chunking_plugins",
    "CHUNKING_REGISTRY",
    "PluginRegistryError",
    "PluginNotFoundError",
    "DuplicatePluginError",
]