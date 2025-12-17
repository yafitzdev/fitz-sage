# fitz/engines/classic_rag/retrieval/runtime/registry.py
"""
Retrieval plugin registry.

This is a thin wrapper around fitz.core.registry.
All the actual logic lives there - this file just provides
backwards-compatible imports.

For new code, prefer importing directly from fitz.core.registry:
    from fitz.core.registry import get_retriever_plugin, available_retriever_plugins
"""

from fitz.core.registry import (
    # Main functions
    get_retriever_plugin,
    available_retriever_plugins,
    # Registry (if needed for advanced use)
    RETRIEVAL_REGISTRY,
    # Errors
    PluginRegistryError,
    PluginNotFoundError,
    DuplicatePluginError,
)

# Backwards compatibility - old name was RETRIEVER_REGISTRY
RETRIEVER_REGISTRY = RETRIEVAL_REGISTRY._registry  # Direct access for legacy code

__all__ = [
    "get_retriever_plugin",
    "available_retriever_plugins",
    "RETRIEVAL_REGISTRY",
    "RETRIEVER_REGISTRY",
    "PluginRegistryError",
    "PluginNotFoundError",
    "DuplicatePluginError",
]