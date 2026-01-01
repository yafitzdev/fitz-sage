# fitz_ai/config/__init__.py
"""
Configuration management for Fitz.

This package provides:
- Default configurations for each engine
- Layered config loading (defaults + user overrides)
- Config merging utilities

Usage:
    from fitz_ai.config import load_engine_config

    # Load merged config (defaults + user overrides)
    config = load_engine_config("fitz_rag")

    # Access values directly - no need for .get() with fallbacks
    chat_plugin = config["chat"]["plugin_name"]  # always exists
"""

from fitz_ai.config.loader import load_engine_config, load_engine_defaults

__all__ = ["load_engine_config", "load_engine_defaults"]
