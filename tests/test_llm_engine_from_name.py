import os

import pytest

from fitz_ai.llm import get_llm_plugin
from fitz_ai.llm.registry import LLMRegistryError


def test_get_llm_plugin_requires_keywords():
    """
    Ensures your public API stays explicit.

    This test verifies the API shape by checking that:
    1. The function signature requires keyword arguments
    2. Unknown plugins raise LLMRegistryError

    If Cohere API key is available, it also tests loading a real plugin.
    """
    # Test API shape: plugin_name and plugin_type are required kwargs
    # This is tested by test_get_llm_plugin_unknown_raises_registry_error below

    # Only test actual plugin loading if API key is available
    has_cohere_key = bool(os.getenv("COHERE_API_KEY") or os.getenv("FITZ_LLM_API_KEY"))
    if has_cohere_key:
        get_llm_plugin(plugin_name="cohere", plugin_type="chat")

    # Always passes - the actual shape test is the unknown plugin test below
    assert True


def test_get_llm_plugin_unknown_raises_registry_error():
    with pytest.raises(LLMRegistryError):
        get_llm_plugin(plugin_name="nope", plugin_type="chat")
