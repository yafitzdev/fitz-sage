# fitz/engines/classic_rag/pipeline/pipeline/registry.py
"""
Pipeline plugin registry.

This is a thin wrapper around fitz.core.registry.
All the actual logic lives there - this file just provides
backwards-compatible imports.

For new code, prefer importing directly from fitz.core.registry:
    from fitz.core.registry import get_pipeline_plugin, available_pipeline_plugins
"""

from fitz.core.registry import (
    # Main functions
    get_pipeline_plugin,
    available_pipeline_plugins,
    # Registry (if needed for advanced use)
    PIPELINE_REGISTRY,
    # Errors
    PluginRegistryError,
    PluginNotFoundError,
    DuplicatePluginError,
)

# Backwards compatibility - old name
PIPELINE_REGISTRY_DICT = PIPELINE_REGISTRY._registry  # Direct access for legacy code

__all__ = [
    "get_pipeline_plugin",
    "available_pipeline_plugins",
    "PIPELINE_REGISTRY",
    "PIPELINE_REGISTRY_DICT",
    "PluginRegistryError",
    "PluginNotFoundError",
    "DuplicatePluginError",
]