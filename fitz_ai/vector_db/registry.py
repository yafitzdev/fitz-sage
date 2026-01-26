# fitz_ai/vector_db/registry.py
"""
Vector DB plugin registry.

Fitz uses PostgreSQL + pgvector for all storage. This registry provides
the pgvector plugin instance.
"""

from __future__ import annotations

from typing import Any, List

from fitz_ai.core.instrumentation import maybe_wrap
from fitz_ai.vector_db.loader import create_vector_db_plugin

# Methods to track for vector DB plugins
_VECTOR_DB_METHODS_TO_TRACK = {"search", "upsert", "delete", "count", "list_collections"}


def get_vector_db_plugin(plugin_name: str = "pgvector", **kwargs: Any) -> Any:
    """
    Get the pgvector plugin instance.

    Fitz uses PostgreSQL + pgvector exclusively for unified storage.
    The plugin_name parameter exists for backwards compatibility but
    only 'pgvector' is supported.

    Args:
        plugin_name: Must be 'pgvector' (default)
        **kwargs: Plugin configuration (mode, connection_string, etc.)

    Returns:
        PgVectorDB plugin instance
    """
    if plugin_name != "pgvector":
        raise ValueError(
            f"Unsupported vector_db plugin: '{plugin_name}'. "
            f"Fitz uses PostgreSQL + pgvector exclusively. Use 'pgvector' or omit the parameter."
        )

    plugin = create_vector_db_plugin(plugin_name, **kwargs)
    # Wrap for instrumentation (only if hooks registered)
    return maybe_wrap(
        plugin,
        layer="vector_db",
        plugin_name=plugin_name,
        methods_to_track=_VECTOR_DB_METHODS_TO_TRACK,
    )


def available_vector_db_plugins() -> List[str]:
    """
    List available vector DB plugins.

    Returns:
        ['pgvector'] - Fitz uses PostgreSQL + pgvector exclusively
    """
    return ["pgvector"]


__all__ = [
    "get_vector_db_plugin",
    "available_vector_db_plugins",
]
