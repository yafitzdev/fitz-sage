# fitz/engines/classic_rag/pipeline/pipeline/registry.py
"""Pipeline plugin registry."""
from __future__ import annotations

from fitz.core.registry import (
    get_pipeline_plugin,
    available_pipeline_plugins,
    PIPELINE_REGISTRY,
    PluginRegistryError,
    PluginNotFoundError,
)

__all__ = [
    "get_pipeline_plugin",
    "available_pipeline_plugins",
    "PIPELINE_REGISTRY",
    "PluginRegistryError",
    "PluginNotFoundError",
]