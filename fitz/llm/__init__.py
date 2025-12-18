# fitz/llm/__init__.py
"""
LLM plugin system for Fitz.

All LLM plugins are YAML-based. Use get_llm_plugin() to get instances.
"""
from __future__ import annotations

from fitz.llm.registry import available_llm_plugins, get_llm_plugin, LLMRegistryError

from fitz.llm.loader import (
    load_plugin,
    list_yaml_plugins,
    clear_cache,
    YAMLPluginError,
    YAMLPluginNotFoundError,
    YAMLPluginValidationError,
)

from fitz.llm.runtime import (
    YAMLPluginBase,
    YAMLChatClient,
    YAMLEmbeddingClient,
    YAMLRerankClient,
    create_yaml_client,
)

__all__ = [
    # Registry (main API)
    "get_llm_plugin",
    "available_llm_plugins",
    "LLMRegistryError",
    # Loader
    "load_plugin",
    "list_yaml_plugins",
    "clear_cache",
    # Errors
    "YAMLPluginError",
    "YAMLPluginNotFoundError",
    "YAMLPluginValidationError",
    # Runtime (for advanced use)
    "YAMLPluginBase",
    "YAMLChatClient",
    "YAMLEmbeddingClient",
    "YAMLRerankClient",
    "create_yaml_client",
]