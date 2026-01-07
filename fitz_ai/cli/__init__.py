# fitz_ai/cli/__init__.py
# =============================================================================
# CRITICAL: Windows Hugging Face symlink fix - MUST be before ANY imports
# =============================================================================
# Windows restricts symlink creation by default, causing Docling model downloads
# to fail with [WinError 1314]. Setting these env vars before huggingface_hub
# is imported makes HF use file copies instead of symlinks.
import os as _os
import sys as _sys

if _sys.platform == "win32":
    _os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    _os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

"""
Fitz CLI - Clean, minimal CLI.

Usage:
    fitz init               # Setup wizard
    fitz ingest ./src       # Ingest documents
    fitz query "question"   # Query knowledge base
    fitz collections        # List/manage collections
    fitz config             # Show configuration
    fitz doctor             # System diagnostics
"""

from fitz_ai.cli.cli import app  # noqa: E402

__all__ = ["app"]
