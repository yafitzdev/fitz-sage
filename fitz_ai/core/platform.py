# fitz_ai/core/platform.py
"""
Platform-specific configuration for Fitz.

Explicit initialization functions - no import-time side effects.
"""

from __future__ import annotations

import os
import sys


def configure_huggingface_windows() -> None:
    """
    Configure Hugging Face Hub for Windows compatibility.

    Windows restricts symlink creation by default, causing model downloads
    to fail with [WinError 1314]. Setting these env vars makes HF use
    file copies instead of symlinks.

    Call this once at application entry point (CLI, API server, etc.)
    before any Hugging Face imports.
    """
    if sys.platform == "win32":
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
