import pytest

from core.llm import get_llm_plugin
from core.llm.registry import LLMRegistryError


def test_vector_db_discovery_in_llm_registry():
    cls = get_llm_plugin(plugin_name="qdrant", plugin_type="vector_db")
    assert isinstance(cls, type)
    assert callable(getattr(cls, "search", None))


def test_vector_db_unknown_raises():
    with pytest.raises(LLMRegistryError):
        get_llm_plugin(plugin_name="nope", plugin_type="vector_db")
