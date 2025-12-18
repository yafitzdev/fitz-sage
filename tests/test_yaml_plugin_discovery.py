# tests/test_yaml_plugin_discovery.py
"""
Tests for YAML plugin discovery.
"""
import pytest


class TestYAMLPluginDiscovery:
    """Test YAML plugin discovery."""

    def test_list_yaml_plugins(self):
        """list_yaml_plugins returns list for each type."""
        from fitz.llm.loader import list_yaml_plugins

        for plugin_type in ("chat", "embedding", "rerank"):
            result = list_yaml_plugins(plugin_type)
            assert isinstance(result, list)

    def test_available_llm_plugins(self):
        """available_llm_plugins includes YAML plugins."""
        from fitz.llm import available_llm_plugins, list_yaml_plugins
        from fitz.core import registry

        # Reset discovery
        registry._LLM_DISCOVERED = False
        registry.LLM_REGISTRY.clear()

        yaml_plugins = list_yaml_plugins("chat")
        all_plugins = available_llm_plugins("chat")

        for name in yaml_plugins:
            assert name in all_plugins

    def test_get_llm_plugin_returns_wrapper(self):
        """get_llm_plugin returns wrapper class for YAML plugins."""
        from fitz.llm import get_llm_plugin, list_yaml_plugins
        from fitz.core import registry

        registry._LLM_DISCOVERED = False
        registry.LLM_REGISTRY.clear()

        yaml_plugins = list_yaml_plugins("chat")
        if not yaml_plugins:
            pytest.skip("No YAML chat plugins")

        plugin_cls = get_llm_plugin(plugin_name=yaml_plugins[0], plugin_type="chat")

        assert hasattr(plugin_cls, "plugin_name")
        assert hasattr(plugin_cls, "plugin_type")
        assert plugin_cls.plugin_name == yaml_plugins[0]
        assert plugin_cls.plugin_type == "chat"

    def test_resolve_llm_plugin(self):
        """resolve_llm_plugin works with YAML plugins."""
        from fitz.core.registry import resolve_llm_plugin
        from fitz.llm.loader import list_yaml_plugins
        from fitz.core import registry

        registry._LLM_DISCOVERED = False
        registry.LLM_REGISTRY.clear()

        yaml_plugins = list_yaml_plugins("chat")
        if not yaml_plugins:
            pytest.skip("No YAML chat plugins")

        plugin_cls = resolve_llm_plugin(
            plugin_type="chat",
            requested_name=yaml_plugins[0],
        )

        assert isinstance(plugin_cls, type)

    def test_invalid_plugin_type_raises(self):
        """Invalid plugin type raises ValueError."""
        from fitz.llm import get_llm_plugin

        with pytest.raises(ValueError):
            get_llm_plugin(plugin_name="test", plugin_type="invalid")

    def test_unknown_plugin_raises(self):
        """Unknown plugin raises LLMRegistryError."""
        from fitz.llm import get_llm_plugin
        from fitz.core.registry import LLMRegistryError

        with pytest.raises(LLMRegistryError):
            get_llm_plugin(plugin_name="__nonexistent__", plugin_type="chat")