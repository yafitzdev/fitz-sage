# fitz_ai/config/loader.py
"""
Configuration loading for Fitz engines.

Usage:
    from fitz_ai.config.loader import load_engine_config

    # Returns Pydantic config model - defaults + user overrides
    config = load_engine_config("fitz_rag")

    # Values are typed and validated
    chat_plugin = config.chat  # "cohere" or "anthropic/claude-sonnet-4"
    top_k = config.top_k  # int, validated >= 1
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


def _load_defaults(engine: str) -> dict[str, Any]:
    """Load package defaults for an engine (unwrapped)."""
    defaults_path = _get_defaults_path(engine)

    if not defaults_path.exists():
        raise FileNotFoundError(
            f"No defaults found for engine '{engine}'. Expected: {defaults_path}"
        )

    with defaults_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Unwrap engine key if present (e.g., fitz_rag: {...})
    if engine in raw:
        defaults = raw[engine]
    else:
        defaults = raw

    logger.debug(f"Loaded defaults for {engine} from {defaults_path}")
    return defaults


def _load_user_config(engine: str) -> dict[str, Any] | None:
    """Load user config for an engine, or None if not found."""
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


def load_engine_config(engine: str):
    """
    Load configuration for an engine.

    Returns a validated Pydantic model with defaults + user overrides.

    Args:
        engine: Engine name (e.g., "fitz_rag")

    Returns:
        Pydantic config model (e.g., FitzRagConfig)

    Raises:
        FileNotFoundError: If no defaults exist
        ImportError: If engine config schema not available
        ValidationError: If config is invalid

    Examples:
        >>> config = load_engine_config("fitz_rag")
        >>> config.chat  # str: "cohere" or "anthropic/claude-sonnet-4"
        >>> config.top_k  # int: validated >= 1
    """
    # Import engine-specific schema
    if engine == "fitz_rag":
        from fitz_ai.engines.fitz_rag.config.schema import FitzRagConfig

        ConfigModel = FitzRagConfig
    else:
        raise ImportError(f"Config schema not available for engine '{engine}'")

    # Load defaults
    defaults = _load_defaults(engine)

    # Load user config (optional)
    user_config = _load_user_config(engine)

    if user_config is not None:
        # Merge user config over defaults
        merged = deep_merge(defaults, user_config)
        logger.debug(f"Merged config for {engine}: defaults + user overrides")
    else:
        merged = defaults
        logger.debug(f"Using defaults only for {engine}")

    # Validate and return Pydantic model
    config_model = ConfigModel(**merged)
    logger.debug(f"Validated config for {engine}")
    return config_model


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
    "deep_merge",
    "get_config_source",
]
