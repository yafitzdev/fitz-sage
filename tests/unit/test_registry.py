# tests/test_registry.py
"""
Tests for the centralized plugin registry system.

Architecture:
- Python-based registries use fitz_ai.core.registry.PluginRegistry
- LLM plugins use fitz_ai.llm.registry (YAML-based)
- Vector DB plugins use fitz_ai.vector_db.registry (YAML-based)
- Chunking plugins use fitz_ai.core.registry.CHUNKING_REGISTRY
"""

import pytest

from fitz_ai.core.registry import PluginNotFoundError, get_chunking_plugin


def test_chunking_registry_returns_correct_plugin():
    """Chunking registry should return the correct plugin class by name."""
    plugin_cls = get_chunking_plugin("simple")
    assert plugin_cls.plugin_name == "simple"


def test_chunking_registry_rejects_unknown_plugin():
    """Chunking registry should raise PluginNotFoundError for unknown plugins."""
    with pytest.raises(PluginNotFoundError):
        get_chunking_plugin("does_not_exist")


def test_registry_error_message_is_helpful():
    """Error message should list available plugins."""
    with pytest.raises(PluginNotFoundError) as exc_info:
        get_chunking_plugin("nonexistent_plugin")

    error_msg = str(exc_info.value)
    assert "nonexistent_plugin" in error_msg
    assert "Available" in error_msg


def test_all_registries_follow_pattern():
    """All plugin registries should follow the same pattern."""
    from fitz_ai.llm.registry import LLMRegistryError, get_llm_plugin
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    # LLM raises LLMRegistryError
    with pytest.raises(LLMRegistryError):
        get_llm_plugin(plugin_name="__fake__", plugin_type="chat")

    # Vector DB raises ValueError (from loader)
    with pytest.raises(ValueError):
        get_vector_db_plugin("__fake__")

    # Chunking raises PluginNotFoundError
    with pytest.raises(PluginNotFoundError):
        get_chunking_plugin("__fake__")
