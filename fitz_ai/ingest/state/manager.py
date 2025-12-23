# fitz_ai/ingest/state/manager.py
"""
State manager for incremental ingestion.

Manages reading and writing of .fitz/ingest.json.

Key responsibilities:
- Load/save state from disk
- Update file entries (mark active, mark deleted)
- Track timestamps

Key non-responsibilities:
- NO vector DB access (that's executor's job)
- NO ingestion logic
- NO diffing logic
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fitz_ai.core.paths import FitzPaths

from .schema import (
    EmbeddingConfig,
    FileEntry,
    FileStatus,
    IngestState,
    RootEntry,
)

logger = logging.getLogger(__name__)


class IngestStateManager:
    """
    Manages .fitz/ingest.json state file.

    This class handles ONLY JSON state - no vector DB operations.

    Usage:
        manager = IngestStateManager()
        manager.load()

        # Update entries
        manager.mark_active("/abs/path/to/file.md", "sha256:abc123", ...)
        manager.mark_deleted("/abs/path/to/deleted.md", "/root/path")

        # Save changes
        manager.save()
    """

    def __init__(self, state_path: Optional[Path] = None) -> None:
        """
        Initialize the state manager.

        Args:
            state_path: Path to ingest.json. If None, uses FitzPaths.ingest_state()
        """
        if state_path is None:
            state_path = FitzPaths.workspace() / "ingest.json"
        self._path = state_path
        self._state: Optional[IngestState] = None
        self._dirty = False

    @property
    def state(self) -> IngestState:
        """Get current state, loading if necessary."""
        if self._state is None:
            self.load()
        assert self._state is not None
        return self._state

    @property
    def path(self) -> Path:
        """Get state file path."""
        return self._path

    def load(self) -> IngestState:
        """
        Load state from disk.

        Creates new state if file doesn't exist.

        Returns:
            Loaded or created IngestState
        """
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self._state = IngestState.model_validate(data)
                logger.debug(f"Loaded ingest state from {self._path}")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load ingest state, creating new: {e}")
                self._state = self._create_new_state()
        else:
            logger.info(f"No ingest state found at {self._path}, creating new")
            self._state = self._create_new_state()

        self._dirty = False
        return self._state

    def save(self) -> None:
        """
        Save state to disk.

        Only writes if there are unsaved changes.
        """
        if self._state is None:
            return

        if not self._dirty:
            logger.debug("State unchanged, skipping save")
            return

        # Update timestamp
        self._state.updated_at = datetime.utcnow()

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically using temp file
        temp_path = self._path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(
                    self._state.model_dump(mode="json"),
                    f,
                    indent=2,
                    default=str,  # Handle datetime serialization
                )
            temp_path.replace(self._path)
            self._dirty = False
            logger.debug(f"Saved ingest state to {self._path}")
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _create_new_state(self) -> IngestState:
        """Create a new empty state."""
        return IngestState(
            project_id=str(uuid.uuid4()),
            updated_at=datetime.utcnow(),
        )

    def _ensure_root(self, root: str) -> RootEntry:
        """Ensure root entry exists, creating if necessary."""
        if root not in self.state.roots:
            self.state.roots[root] = RootEntry()
            self._dirty = True
        return self.state.roots[root]

    def mark_active(
        self,
        file_path: str,
        root: str,
        content_hash: str,
        ext: str,
        size_bytes: int,
        mtime_epoch: float,
    ) -> None:
        """
        Mark a file as active in state.

        Called after successful ingestion or when file is unchanged.

        Args:
            file_path: Absolute path to file
            root: Absolute path to root directory
            content_hash: SHA-256 content hash
            ext: File extension (e.g., ".md")
            size_bytes: File size in bytes
            mtime_epoch: File modification time as Unix epoch
        """
        root_entry = self._ensure_root(root)

        root_entry.files[file_path] = FileEntry(
            ext=ext,
            size_bytes=size_bytes,
            mtime_epoch=mtime_epoch,
            content_hash=content_hash,
            status=FileStatus.ACTIVE,
            last_seen_at=datetime.utcnow(),
        )
        root_entry.last_run_at = datetime.utcnow()
        self._dirty = True

    def mark_deleted(self, file_path: str, root: str) -> None:
        """
        Mark a file as deleted in state.

        Called when a file is no longer present on disk.

        Args:
            file_path: Absolute path to file
            root: Absolute path to root directory
        """
        root_entry = self.state.roots.get(root)
        if root_entry is None:
            return

        file_entry = root_entry.files.get(file_path)
        if file_entry is None:
            return

        if file_entry.status != FileStatus.DELETED:
            file_entry.status = FileStatus.DELETED
            file_entry.last_seen_at = datetime.utcnow()
            self._dirty = True

    def get_known_paths(self, root: str) -> set[str]:
        """
        Get all known file paths for a root.

        Returns both active and deleted paths.
        """
        return self.state.get_known_paths(root)

    def get_active_paths(self, root: str) -> set[str]:
        """
        Get all active (non-deleted) file paths for a root.
        """
        return self.state.get_active_paths(root)

    def get_file_entry(self, root: str, file_path: str) -> Optional[FileEntry]:
        """Get file entry if it exists."""
        return self.state.get_file_entry(root, file_path)

    def set_embedding_config(
        self,
        provider: str,
        model: str,
        **kwargs,
    ) -> None:
        """
        Set the current embedding configuration.

        Args:
            provider: Embedding provider name
            model: Model name
            **kwargs: Additional config (dimension, normalize)
        """
        self.state.embedding = EmbeddingConfig.create(provider, model, **kwargs)
        self._dirty = True

    def get_embedding_id(self) -> str:
        """Get the current embedding ID."""
        return self.state.get_embedding_id()

    def get_parser_id(self, ext: str) -> str:
        """Get parser ID for an extension."""
        return self.state.get_parser_id(ext)

    def get_chunker_id(self, ext: str) -> str:
        """Get chunker ID for an extension."""
        return self.state.get_chunker_id(ext)


__all__ = ["IngestStateManager"]