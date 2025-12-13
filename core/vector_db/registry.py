# core/vector_db/registry.py
from __future__ import annotations

from typing import Any, Type

from core.llm.registry import get_llm_plugin


def get_vector_db_plugin(plugin_name: str) -> Type[Any]:
    return get_llm_plugin(plugin_name=plugin_name, plugin_type="vector_db")
