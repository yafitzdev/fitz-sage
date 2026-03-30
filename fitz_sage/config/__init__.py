# fitz_sage/config/__init__.py
"""
Configuration management for Fitz.

This package provides:
- Default configurations for each engine
- Layered config loading (defaults + user overrides)
- Config merging utilities

Usage:
    from fitz_sage.config import load_engine_config

    # Load merged config (defaults + user overrides)
    config = load_engine_config("fitz_krag")

    # Access typed and validated config
    chat_plugin = config.chat_smart  # "ollama/qwen2.5:14b"
"""

from fitz_sage.config.loader import load_engine_config

__all__ = ["load_engine_config"]
