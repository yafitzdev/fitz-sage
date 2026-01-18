# fitz_ai/core/paths/workspace.py
"""
Workspace path management - foundation for all other paths.

The workspace is the .fitz directory in the current working directory,
or an override set for testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class WorkspaceManager:
    """Internal workspace state manager."""

    _workspace_override: Optional[Path] = None

    @classmethod
    def set_workspace(cls, path: Optional[str | Path]) -> None:
        """
        Override the workspace root.

        Useful for testing or running multiple isolated instances.
        Pass None to reset to default (CWD).
        """
        if path is None:
            cls._workspace_override = None
        else:
            cls._workspace_override = Path(path)

    @classmethod
    def reset(cls) -> None:
        """Reset to default workspace (CWD). Useful in tests."""
        cls._workspace_override = None

    @classmethod
    def workspace(cls) -> Path:
        """
        The .fitz workspace directory.

        Default: {CWD}/.fitz/
        """
        if cls._workspace_override is not None:
            return cls._workspace_override
        return Path.cwd() / ".fitz"

    @classmethod
    def ensure_workspace(cls) -> Path:
        """Get workspace path and create it if it doesn't exist."""
        path = cls.workspace()
        path.mkdir(parents=True, exist_ok=True)
        return path


def workspace() -> Path:
    """The .fitz workspace directory."""
    return WorkspaceManager.workspace()


def ensure_workspace() -> Path:
    """Get workspace path and create it if it doesn't exist."""
    return WorkspaceManager.ensure_workspace()


def set_workspace(path: Optional[str | Path]) -> None:
    """Override the workspace root."""
    WorkspaceManager.set_workspace(path)


def reset() -> None:
    """Reset to default workspace (CWD)."""
    WorkspaceManager.reset()
