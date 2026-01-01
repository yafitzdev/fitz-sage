# fitz_ai/config/loader.py
"""
Layered configuration loading for Fitz engines.

This module implements the config merge strategy:
    1. Package defaults (fitz_ai/config/defaults/<engine>.yaml) - always loaded
    2. User config (.fitz/config/<engine>.yaml) - overrides defaults

The result is a complete config where every value is guaranteed to exist.
CLI code never needs fallback logic - just read the values.

Usage:
    from fitz_ai.config.loader import load_engine_config

    # Returns merged config - defaults + user overrides
    config = load_engine_config("fitz_rag")

    # Values are guaranteed to exist
    chat_plugin = config["chat"]["plugin_name"]
    embedding_model = config["embedding"]["kwargs"]["model"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Engine Paths
# =============================================================================


def _get_engine_config_dir(engine: str) -> Path:
    """Get the config directory for an engine."""
    return Path(__file__).parent.parent / "engines" / engine / "config"


def _get_defaults_path(engine: str) -> Path:
    """Get the path to the default config for an engine."""
    return _get_engine_config_dir(engine) / "default.yaml"


# =============================================================================
# Deep Merge
# =============================================================================


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.

    Values from `override` take precedence over `base`.
    Nested dicts are merged recursively.
    Lists are replaced entirely (not merged).

    Args:
        base: Base dictionary (defaults)
        override: Override dictionary (user config)

    Returns:
        New merged dictionary

    Examples:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"c": 10}}
        >>> deep_merge(base, override)
        {"a": 1, "b": {"c": 10, "d": 3}}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override the value (including lists)
            result[key] = value

    return result


# =============================================================================
# Loading Functions
# =============================================================================


def load_engine_defaults(engine: str) -> dict[str, Any]:
    """
    Load package defaults for an engine.

    Args:
        engine: Engine name (e.g., "fitz_rag")

    Returns:
        Default configuration dictionary (unwrapped from engine key)

    Raises:
        FileNotFoundError: If no defaults exist for this engine
    """
    defaults_path = _get_defaults_path(engine)

    if not defaults_path.exists():
        raise FileNotFoundError(
            f"No defaults found for engine '{engine}'. "
            f"Expected: {defaults_path}"
        )

    with defaults_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Engine defaults are nested under engine name (e.g., fitz_rag: {...})
    # User configs are flat. Unwrap to match user config structure.
    if engine in raw:
        defaults = raw[engine]
    else:
        defaults = raw

    logger.debug(f"Loaded defaults for {engine} from {defaults_path}")
    return defaults


def load_user_config(engine: str) -> dict[str, Any] | None:
    """
    Load user configuration for an engine.

    Looks in .fitz/config/<engine>.yaml

    Args:
        engine: Engine name (e.g., "fitz_rag")

    Returns:
        User configuration dictionary, or None if no user config exists
    """
    user_path = FitzPaths.engine_config(engine)

    if not user_path.exists():
        logger.debug(f"No user config for {engine} at {user_path}")
        return None

    try:
        with user_path.open("r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        logger.debug(f"Loaded user config for {engine} from {user_path}")
        return user_config
    except Exception as e:
        logger.warning(f"Failed to load user config from {user_path}: {e}")
        return None


def load_engine_config(engine: str) -> dict[str, Any]:
    """
    Load complete configuration for an engine.

    Merges package defaults with user overrides.
    The result is guaranteed to have all required values.

    Merge order:
        1. Package defaults (fitz_ai/config/defaults/<engine>.yaml)
        2. User config (.fitz/config/<engine>.yaml) - overrides defaults

    Args:
        engine: Engine name (e.g., "fitz_rag")

    Returns:
        Complete merged configuration dictionary

    Examples:
        >>> config = load_engine_config("fitz_rag")
        >>> config["chat"]["plugin_name"]  # always exists
        'cohere'
        >>> config["retrieval"]["top_k"]  # always exists
        5
    """
    # Load defaults (required)
    defaults = load_engine_defaults(engine)

    # Load user config (optional)
    user_config = load_user_config(engine)

    if user_config is None:
        logger.debug(f"Using defaults only for {engine}")
        return defaults

    # Merge: user overrides defaults
    merged = deep_merge(defaults, user_config)
    logger.debug(f"Merged config for {engine}: defaults + user overrides")
    return merged


def get_config_source(engine: str) -> str:
    """
    Get a description of where config is loaded from.

    Useful for CLI display.

    Args:
        engine: Engine name

    Returns:
        Human-readable source description
    """
    user_path = FitzPaths.engine_config(engine)

    if user_path.exists():
        return f"{user_path} (overriding defaults)"
    else:
        defaults_path = _get_defaults_path(engine)
        return f"{defaults_path} (package defaults)"


__all__ = [
    "load_engine_config",
    "load_engine_defaults",
    "load_user_config",
    "deep_merge",
    "get_config_source",
]
