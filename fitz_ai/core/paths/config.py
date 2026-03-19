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
