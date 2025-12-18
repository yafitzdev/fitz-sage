# fitz/llm/registry.py
"""
Central LLM plugin registry.

All LLM plugins (chat, embedding, rerank) are YAML-based.
This module re-exports registry functions from fitz.core.registry.

Design principle: NO SILENT FALLBACK
- If user configures "cohere", they get cohere or an error
- No magic substitution
"""
from __future__ import annotations

from fitz.core.registry import (
    LLM_REGISTRY,
    LLMRegistryError,
    PluginNotFoundError,
    PluginRegistryError,
    available_llm_plugins,
    get_llm_plugin,
    resolve_llm_plugin,
)

# Type alias
LLMPluginType = str  # "chat" | "embedding" | "rerank"

__all__ = [
    # Functions
    "get_llm_plugin",
    "available_llm_plugins",
    "resolve_llm_plugin",
    # Registry
    "LLM_REGISTRY",
    # Errors
    "LLMRegistryError",
    "PluginRegistryError",
    "PluginNotFoundError",
    # Types
    "LLMPluginType",
]