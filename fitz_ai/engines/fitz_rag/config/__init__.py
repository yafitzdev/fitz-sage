# fitz_ai/engines/fitz_rag/config/__init__.py
"""
Configuration for Fitz RAG engine.

This package provides:
- FitzRagConfig: The main configuration schema
- load_config: Configuration loader function
- Supporting schemas (PluginConfig, RetrievalConfig, IngestConfig, etc.)

Usage:
    >>> from fitz_ai.engines.fitz_rag.config import load_config, FitzRagConfig
    >>> config = load_config("config.yaml")
"""

from .loader import (
    DEFAULT_CONFIG_PATH,
    get_default_config_path,
    get_user_config_path,
    load_config,
    load_config_dict,
)
from .schema import (  # Main config; RAG sub-configs; Ingestion configs
    ChunkingRouterConfig,
    ExtensionChunkerConfig,
    FitzRagConfig,
    IngestConfig,
    IngesterConfig,
    LoggingConfig,
    PluginConfig,
    RerankConfig,
    RetrievalConfig,
    RGSConfig,
    StructuredConfig,
)

__all__ = [
    # Main config
    "FitzRagConfig",
    "load_config",
    "load_config_dict",
    "get_default_config_path",
    "get_user_config_path",
    # RAG sub-configs
    "PluginConfig",
    "RetrievalConfig",
    "RerankConfig",
    "RGSConfig",
    "LoggingConfig",
    "StructuredConfig",
    # Ingestion configs
    "IngestConfig",
    "IngesterConfig",
    "ChunkingRouterConfig",
    "ExtensionChunkerConfig",
    # Constants
    "DEFAULT_CONFIG_PATH",
]
