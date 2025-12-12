# core/vector_db/registry.py
from __future__ import annotations

from typing import Type

from core.vector_db.base import VectorDBPlugin
from core.llm.registry import get_llm_plugin


def get_vector_db_plugin(plugin_name: str) -> Type[VectorDBPlugin]:
    """
    Return the vector DB plugin class for the given plugin name.

    Single source of truth: central plugin registry.
    """
    return get_llm_plugin(plugin_name=plugin_name, plugin_type="vector_db")  # type: ignore[return-value]
