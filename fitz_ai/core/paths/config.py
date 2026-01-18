# fitz_ai/core/paths/config.py
"""Configuration file paths."""

from __future__ import annotations

from pathlib import Path

from .workspace import workspace


def config() -> Path:
    """
    Default config file path.

    Location: {workspace}/config.yaml
    """
    return workspace() / "config.yaml"


def config_dir() -> Path:
    """
    Config directory for engine-specific config files.

    Location: {workspace}/config/
    """
    return workspace() / "config"


def engine_config(engine_name: str) -> Path:
    """
    Engine-specific config file path.

    Location: {workspace}/config/{engine_name}.yaml
    """
    return config_dir() / f"{engine_name}.yaml"


def ensure_config_dir() -> Path:
    """Get config directory and create it if it doesn't exist."""
    path = config_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
