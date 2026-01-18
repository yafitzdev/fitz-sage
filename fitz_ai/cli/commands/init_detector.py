# fitz_ai/cli/commands/init_detector.py
"""
System detection and plugin discovery for init wizard.
"""

from __future__ import annotations

from fitz_ai.cli.services import InitService

# Service instance for business logic
_init_service = InitService()


def load_default_config() -> dict:
    """Load the default configuration from default.yaml."""
    return _init_service.load_default_config()


def detect_system():
    """Detect all available services and API keys."""
    return _init_service.detect_system()


def filter_available_plugins(plugins: list[str], plugin_type: str, system) -> list[str]:
    """Filter plugins to only those that are available."""
    return _init_service.filter_available_plugins(plugins, plugin_type, system)
