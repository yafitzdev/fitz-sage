# fitz_sage/engines/fitz_krag/config/__init__.py
"""Configuration for Fitz KRAG engine."""

from pathlib import Path

from .schema import FitzKragConfig


def get_default_config_path() -> Path:
    """Get the path to the default configuration file."""
    return Path(__file__).parent / "default.yaml"


__all__ = [
    "FitzKragConfig",
    "get_default_config_path",
]
