# fitz_ai/engines/clara/config/loader.py
"""
Configuration loader for CLaRa engine.

Usage:
    from fitz_ai.engines.clara.config.loader import load_clara_config
    config = load_clara_config()  # Returns defaults
    config = load_clara_config("my_config.yaml")  # Loads from file
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from fitz_ai.core.config import (
    ConfigError,
    ConfigNotFoundError,
    load_config,
)
from fitz_ai.engines.clara.config.schema import ClaraConfig


def load_clara_config(config_path: Optional[Union[str, Path]] = None) -> ClaraConfig:
    """
    Load CLaRa configuration from a YAML file.

    Args:
        config_path: Path to YAML config file. If None, returns defaults.

    Returns:
        ClaraConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        return ClaraConfig()

    try:
        return load_config(config_path, schema=ClaraConfig)
    except ConfigNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except ConfigError as e:
        raise RuntimeError(f"Failed to load CLaRa config: {e}") from e


__all__ = [
    "load_clara_config",
]
