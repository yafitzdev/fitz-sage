"""
Configuration presets for Fitz.

Provides pre-configured setups for common use cases:
- local: Offline development with Ollama
- dev: Development with cloud APIs (cost-effective models)
- production: Production-ready with best models

Usage:
    from fitz.core.config.presets import get_preset

    config = get_preset("local")
    # Returns dict ready to use with pipeline
"""

# Available configuration presets
PRESETS = {
    "local": {
        "vector_db": {
            "plugin_name": "qdrant",
            "kwargs": {"host": "localhost", "port": 6333}
        },
        "llm": {"plugin_name": "local"},
        "embedding": {"plugin_name": "local"},
        "retriever": {
            "plugin_name": "dense",
            "collection": "test_collection",
            "top_k": 5
        },
        "rerank": {
            "enabled": False,
            "plugin_name": "local"
        },
    },
    "dev": {
        "vector_db": {
            "plugin_name": "qdrant",
            "kwargs": {"host": "localhost", "port": 6333}
        },
        "llm": {
            "plugin_name": "openai",
            "kwargs": {"model": "gpt-4o-mini"}
        },
        "embedding": {"plugin_name": "openai"},
        "retriever": {
            "plugin_name": "dense",
            "collection": "dev_collection",
            "top_k": 5
        },
        "rerank": {
            "enabled": True,
            "plugin_name": "cohere"
        },
    },
    "production": {
        "vector_db": {
            "plugin_name": "qdrant",
            "kwargs": {"url": "https://your-cluster.qdrant.io", "api_key": "${QDRANT_API_KEY}"}
        },
        "llm": {"plugin_name": "anthropic"},
        "embedding": {"plugin_name": "cohere"},
        "retriever": {
            "plugin_name": "dense",
            "collection": "production_collection",
            "top_k": 10
        },
        "rerank": {
            "enabled": True,
            "plugin_name": "cohere"
        },
    },
}


def get_preset(name: str) -> dict:
    """
    Get a configuration preset by name.

    Args:
        name: Preset name ("local", "dev", or "production")

    Returns:
        Configuration dictionary ready to use with RAG pipeline

    Raises:
        ValueError: If preset name is not recognized

    Examples:
        >>> config = get_preset("local")
        >>> config["llm"]["plugin_name"]
        'local'

        >>> config = get_preset("dev")
        >>> config["llm"]["kwargs"]["model"]
        'gpt-4o-mini'

        >>> get_preset("invalid")
        Traceback (most recent call last):
        ...
        ValueError: Unknown preset: 'invalid'. Available presets: local, dev, production
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(
            f"Unknown preset: {name!r}. Available presets: {available}"
        )

    # Return a deep copy to prevent mutation of the original preset
    import copy
    return copy.deepcopy(PRESETS[name])