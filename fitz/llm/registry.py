# fitz/llm/registry.py
"""
LLM plugin registry.

This is a thin wrapper around fitz.core.registry.
All the actual logic lives there - this file just provides
backwards-compatible imports.

For new code, prefer importing directly from fitz.core.registry:
    from fitz.core.registry import get_llm_plugin, available_llm_plugins
"""

from fitz.core.registry import (
    # Main functions
    get_llm_plugin,
    available_llm_plugins,
    resolve_llm_plugin,
    # Registries (if needed for advanced use)
    CHAT_REGISTRY,
    EMBEDDING_REGISTRY,
    RERANK_REGISTRY,
    # Errors
    PluginRegistryError,
    PluginNotFoundError,
    DuplicatePluginError,
)

# Backwards compatibility alias
LLMRegistryError = PluginRegistryError

__all__ = [
    "get_llm_plugin",
    "available_llm_plugins",
    "resolve_llm_plugin",
    "CHAT_REGISTRY",
    "EMBEDDING_REGISTRY",
    "RERANK_REGISTRY",
    "PluginRegistryError",
    "PluginNotFoundError",
    "DuplicatePluginError",
    "LLMRegistryError",
]