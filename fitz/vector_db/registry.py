# fitz/vector_db/registry.py
"""
Vector DB plugin registry.

This is a thin wrapper around fitz.core.registry.
All the actual logic lives there - this file just provides
backwards-compatible imports.

For new code, prefer importing directly from fitz.core.registry:
    from fitz.core.registry import get_vector_db_plugin, available_vector_db_plugins
"""

from fitz.core.registry import (
    # Main functions
    get_vector_db_plugin,
    available_vector_db_plugins,
    resolve_vector_db_plugin,
    # Registry (if needed for advanced use)
    VECTOR_DB_REGISTRY,
    # Errors
    PluginRegistryError,
    PluginNotFoundError,
    DuplicatePluginError,
)

# Backwards compatibility alias
VectorDBRegistryError = PluginRegistryError

__all__ = [
    "get_vector_db_plugin",
    "available_vector_db_plugins",
    "resolve_vector_db_plugin",
    "VECTOR_DB_REGISTRY",
    "PluginRegistryError",
    "PluginNotFoundError",
    "DuplicatePluginError",
    "VectorDBRegistryError",
]