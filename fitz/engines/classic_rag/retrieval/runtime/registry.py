# fitz/engines/classic_rag/retrieval/runtime/registry.py
"""Retriever plugin registry."""
from __future__ import annotations

from fitz.core.registry import (
    get_retriever_plugin,
    available_retriever_plugins,
    RETRIEVER_REGISTRY,
    PluginRegistryError,
    PluginNotFoundError,
)

__all__ = [
    "get_retriever_plugin",
    "available_retriever_plugins",
    "RETRIEVER_REGISTRY",
    "PluginRegistryError",
    "PluginNotFoundError",
]