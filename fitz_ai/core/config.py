# fitz_ai/core/config.py
"""
Generic configuration loading utilities for Fitz.

This module provides paradigm-agnostic YAML loading and validation utilities.
Engine-specific config loading lives in each engine's package.

Usage:
    from fitz_ai.core.config import load_yaml, load_config, ConfigError

    # Load raw YAML
    data = load_yaml("config.yaml")

    # Load and validate with a schema
    from fitz_ai.engines.fitz_rag.config.schema import FitzRagConfig
    config = load_config("config.yaml", FitzRagConfig)

    # For engine-specific loading, use the engine's loader:
    from fitz_ai.engines.fitz_rag.config.loader import load_config as load_rag_config

Design principles:
    - Generic utilities only - no engine imports
    - Schema validation is separate from loading
    - Clear error messages with file paths
    - FitzPaths integration for standard locations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml

from fitz_ai.core.paths import FitzPaths

logger = logging.getLogger(__name__)

# Type variable for config classes
T = TypeVar("T")


# =============================================================================
# Errors
# =============================================================================


class ConfigError(Exception):
    """Base error for configuration issues."""

    def __init__(self, message: str, path: Optional[Path] = None):
        self.path = path
        if path:
            message = f"{message} (file: {path})"
        super().__init__(message)


class ConfigNotFoundError(ConfigError):
    """Raised when a config file doesn't exist."""

    pass


class ConfigParseError(ConfigError):
    """Raised when YAML parsing fails."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when config doesn't match schema."""

    pass


# =============================================================================
# Core Loading Functions
# =============================================================================


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file and return as dictionary.

    This is the lowest-level loading function. For most use cases,
    prefer load_config() which also validates.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary of config data

    Raises:
        ConfigNotFoundError: If file doesn't exist
        ConfigParseError: If YAML is invalid
    """
    p = Path(path)

    if not p.exists():
        raise ConfigNotFoundError("Config file not found", path=p)

    if p.is_dir():
        raise ConfigError("Config path is a directory, not a file", path=p)

    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}", path=p) from e
    except Exception as e:
        raise ConfigParseError(f"Failed to read config: {e}", path=p) from e

    if not isinstance(data, dict):
        raise ConfigParseError("Config root must be a mapping (dict)", path=p)

    logger.debug(f"Loaded config from {p}")
    return data


def load_config(
    path: Union[str, Path],
    schema: Type[T],
) -> T:
    """
    Load and validate a configuration file.

    Args:
        path: Path to config file (required).
        schema: Pydantic model or dataclass to validate against (required).

    Returns:
        Validated config object

    Raises:
        ConfigNotFoundError: If file doesn't exist
        ConfigParseError: If YAML is invalid
        ConfigValidationError: If config doesn't match schema

    Examples:
        # Load RAG config with explicit schema
        from fitz_ai.engines.fitz_rag.config.schema import FitzRagConfig
        config = load_config("config.yaml", FitzRagConfig)

        # For engine-specific defaults, use the engine's loader:
        from fitz_ai.engines.fitz_rag.config.loader import load_config as load_rag_config
        config = load_rag_config()  # Loads default.yaml
    """
    resolved_path = Path(path)

    # Load raw YAML
    data = load_yaml(resolved_path)

    # Validate with schema
    return _validate_config(data, schema, resolved_path)


def load_config_dict(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration as raw dictionary (no validation).

    Useful for:
    - Inspecting config before deciding which schema to use
    - Advanced use cases where raw access is needed
    - CLI commands that just display config

    Args:
        path: Path to config file

    Returns:
        Raw config dictionary
    """
    if path is None:
        path = FitzPaths.config()
    return load_yaml(path)


# =============================================================================
# Schema Validation
# =============================================================================


def _validate_config(
    data: Dict[str, Any],
    schema: Type[T],
    path: Path,
) -> T:
    """
    Validate config data against a schema.

    Supports both Pydantic models and dataclasses with from_dict().
    """
    try:
        # Try Pydantic model_validate first
        if hasattr(schema, "model_validate"):
            return schema.model_validate(data)

        # Try from_dict (for dataclasses)
        if hasattr(schema, "from_dict"):
            return schema.from_dict(data)

        # Try direct instantiation
        return schema(**data)

    except Exception as e:
        raise ConfigValidationError(
            f"Config validation failed: {e}",
            path=path,
        ) from e


# =============================================================================
# Config Writing (for fitz init)
# =============================================================================


def save_config(
    data: Union[Dict[str, Any], Any],
    path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save configuration to a YAML file.

    Args:
        data: Config data (dict or object with to_dict/model_dump)
        path: Path to save to. If None, uses FitzPaths.config()

    Returns:
        Path where config was saved
    """
    if path is None:
        path = FitzPaths.config()

    p = Path(path)

    # Ensure parent directory exists
    p.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict if needed
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif hasattr(data, "to_dict"):
        data = data.to_dict()
    elif not isinstance(data, dict):
        raise ConfigError(f"Cannot save config of type {type(data)}")

    # Write YAML
    with p.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {p}")
    return p
