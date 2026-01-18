# fitz_ai/cli/commands/init.py
"""
Init command - Interactive setup wizard.

Usage:
    fitz init              # Interactive wizard
    fitz init -y           # Auto-detect and use defaults
    fitz init --show       # Preview config without saving

This module is the entry point for the init command.
Implementation is split across:
    - init_detector.py: System detection and plugin discovery
    - init_models.py: Model selection helpers
    - init_config.py: YAML config generation
    - init_wizard.py: Main wizard orchestration
"""

from __future__ import annotations

from .init_wizard import command

__all__ = ["command"]
