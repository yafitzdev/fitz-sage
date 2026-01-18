# fitz_ai/engines/fitz_rag/config/__init__.py
"""
Configuration for Fitz RAG engine.

This package provides:
- FitzRagConfig: The main configuration schema (Pydantic model)
- Flat, simplified schema with string plugin specs

Usage:
    >>> from fitz_ai.config import load_engine_config
    >>> config = load_engine_config("fitz_rag")
    >>> config.chat  # "cohere" or "anthropic/claude-sonnet-4"
"""

from pathlib import Path

from .schema import (
    ChunkingRouterConfig,
    ExtensionChunkerConfig,
    FitzRagConfig,
    PluginKwargs,
)


def get_default_config_path() -> Path:
    """Get the path to the default configuration file."""
    return Path(__file__).parent / "default.yaml"


__all__ = [
    "FitzRagConfig",
    "ChunkingRouterConfig",
    "ExtensionChunkerConfig",
    "PluginKwargs",
    "get_default_config_path",
]
