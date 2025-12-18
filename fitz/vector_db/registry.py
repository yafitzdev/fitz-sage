# fitz/vector_db/registry.py

from __future__ import annotations

from typing import Any, List

from fitz.vector_db.loader import create_vector_db_plugin, load_vector_db_spec
from pathlib import Path


def get_vector_db_plugin(plugin_name: str, **kwargs) -> Any:
    """
    Get a vector DB plugin by name.

    Args:
        plugin_name: Name of the plugin (e.g., 'qdrant', 'pinecone', 'local-faiss')
        **kwargs: Plugin configuration (host, port, etc.)

    Returns:
        Vector DB plugin instance

    Raises:
        ValueError: If plugin not found

    Examples:
        >>> db = get_vector_db_plugin('qdrant', host='localhost', port=6333)
        >>> db = get_vector_db_plugin('pinecone', index_name='my-index', project_id='abc')
    """
    return create_vector_db_plugin(plugin_name, **kwargs)


def available_vector_db_plugins() -> List[str]:
    """
    List available vector DB plugins.

    Returns:
        List of plugin names

    Examples:
        >>> available_vector_db_plugins()
        ['qdrant', 'pinecone', 'local-faiss']
    """
    # Scan for YAML files in plugins directory
    plugins_dir = Path(__file__).parent / "plugins"

    if not plugins_dir.exists():
        return []

    yaml_files = list(plugins_dir.glob("*.yaml"))
    return sorted([f.stem for f in yaml_files])


# Backwards compatibility aliases
resolve_vector_db_plugin = get_vector_db_plugin

__all__ = [
    "get_vector_db_plugin",
    "available_vector_db_plugins",
    "resolve_vector_db_plugin",
]