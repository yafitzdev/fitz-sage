# fitz_ai/ingest/diff/differ.py
"""
Diff computation for incremental ingestion.

Computes the action plan by comparing:
1. Scanned files from disk
2. Vector DB (authoritative source)
3. State file (for deletion tracking only)

Key rule from spec:
- Vector DB existence check is MANDATORY for skip decisions
- State is only for path tracking and deletion detection

This module ONLY computes actions - it does NOT execute them.
Execution is handled by the executor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Protocol, Set, runtime_checkable

from .scanner import ScannedFile

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorDBReader(Protocol):
    """
    Protocol for reading from vector DB.

    Must be implemented by vector DB plugins to support diff ingestion.
    """

    def has_content_hash(
        self,
        collection: str,
        content_hash: str,
        parser_id: str,
        chunker_id: str,
        embedding_id: str,
    ) -> bool:
        """
        Check if vectors exist for a given content hash + config.

        This is the authoritative check per spec §5:
        "Even if .fitz/ingest.json is deleted/modified, ingestion must still
        correctly skip unchanged content by checking the vector DB."

        Args:
            collection: Vector DB collection name
            content_hash: SHA-256 hash of file content
            parser_id: Parser identifier (e.g., "md.v1")
            chunker_id: Chunker identifier (e.g., "tokens_800_120")
            embedding_id: Embedding identifier (e.g., "openai:text-embedding-3-small")

        Returns:
            True if vectors exist with is_deleted=false, False otherwise
        """
        ...


@runtime_checkable
class StateReader(Protocol):
    """
    Protocol for reading ingestion state.

    Used only for deletion detection - NOT for skip decisions.
    """

    def get_active_paths(self, root: str) -> Set[str]:
        """Get all active (non-deleted) file paths for a root."""
        ...

    def get_parser_id(self, ext: str) -> str:
        """Get parser ID for an extension."""
        ...

    def get_chunker_id(self, ext: str) -> str:
        """Get chunker ID for an extension."""
        ...

    def get_embedding_id(self) -> str:
        """Get the current embedding ID."""
        ...


@dataclass(frozen=True)
class FileCandidate:
    """
    A file candidate for ingestion.

    Contains all information needed to ingest or skip a file.
    """
    path: str  # Absolute path
    root: str  # Absolute path of scan root
    ext: str  # Extension (e.g., ".md")
    size_bytes: int  # File size
    mtime_epoch: float  # Modification time
    content_hash: str  # SHA-256 content hash
    parser_id: str  # Parser ID for this file type
    chunker_id: str  # Chunker ID for this file type
    embedding_id: str  # Embedding ID

    @classmethod
    def from_scanned(
        cls,
        scanned: ScannedFile,
        parser_id: str,
        chunker_id: str,
        embedding_id: str,
    ) -> "FileCandidate":
        """Create from a ScannedFile with config IDs."""
        return cls(
            path=scanned.path,
            root=scanned.root,
            ext=scanned.ext,
            size_bytes=scanned.size_bytes,
            mtime_epoch=scanned.mtime_epoch,
            content_hash=scanned.content_hash,
            parser_id=parser_id,
            chunker_id=chunker_id,
            embedding_id=embedding_id,
        )


@dataclass
class DiffResult:
    """
    Result of diff computation.

    Contains three lists of actions:
    - to_ingest: Files that need to be ingested (new or changed)
    - to_skip: Files that already exist in vector DB (unchanged)
    - to_mark_deleted: File paths that no longer exist on disk
    """
    to_ingest: List[FileCandidate] = field(default_factory=list)
    to_skip: List[FileCandidate] = field(default_factory=list)
    to_mark_deleted: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"ingest={len(self.to_ingest)}, "
            f"skip={len(self.to_skip)}, "
            f"delete={len(self.to_mark_deleted)}"
        )


class Differ:
    """
    Computes diff between disk state and vector DB.

    IMPORTANT: This class ONLY computes actions - it does NOT execute them.

    The diff algorithm (per spec §7.2):
    1. For each scanned file, check if vectors exist in vector DB
       with matching (content_hash, parser_id, chunker_id, embedding_id)
    2. If vectors exist with is_deleted=false → skip
    3. If vectors don't exist → ingest
    4. For paths in state but not on disk → mark deleted

    Usage:
        differ = Differ(vector_db_reader, state_reader, collection)
        result = differ.compute_diff(scan_result.files)

        # Result contains action plan, NOT executed actions
        for file in result.to_ingest:
            # These files need to be ingested
            pass
    """

    def __init__(
        self,
        vector_db_reader: VectorDBReader,
        state_reader: StateReader,
        collection: str,
    ) -> None:
        """
        Initialize the differ.

        Args:
            vector_db_reader: Reader for checking vector DB existence
            state_reader: Reader for state (deletion detection only)
            collection: Vector DB collection name
        """
        self._vdb = vector_db_reader
        self._state = state_reader
        self._collection = collection

    def compute_diff(
        self,
        scanned_files: List[ScannedFile],
        force: bool = False,
    ) -> DiffResult:
        """
        Compute the diff action plan.

        Args:
            scanned_files: Files from the scanner
            force: If True, skip vector DB check and ingest all files

        Returns:
            DiffResult with action plan
        """
        result = DiffResult()

        # Get embedding ID once
        embedding_id = self._state.get_embedding_id()

        # Track which paths we've seen for deletion detection
        seen_paths: Set[str] = set()
        root: str | None = None

        for scanned in scanned_files:
            # Track root for deletion detection
            if root is None:
                root = scanned.root
            seen_paths.add(scanned.path)

            # Get config IDs for this file type
            parser_id = self._state.get_parser_id(scanned.ext)
            chunker_id = self._state.get_chunker_id(scanned.ext)

            candidate = FileCandidate.from_scanned(
                scanned,
                parser_id=parser_id,
                chunker_id=chunker_id,
                embedding_id=embedding_id,
            )

            if force:
                # Force mode: ingest everything
                result.to_ingest.append(candidate)
                continue

            # CRITICAL: Check vector DB, NOT state
            # This is the authoritative check per spec §5
            try:
                exists = self._vdb.has_content_hash(
                    collection=self._collection,
                    content_hash=scanned.content_hash,
                    parser_id=parser_id,
                    chunker_id=chunker_id,
                    embedding_id=embedding_id,
                )
            except Exception as e:
                # If vector DB check fails, assume we need to ingest
                logger.warning(f"Vector DB check failed for {scanned.path}: {e}")
                exists = False

            if exists:
                result.to_skip.append(candidate)
            else:
                result.to_ingest.append(candidate)

        # Deletion detection: compare state paths with scanned paths
        if root is not None:
            previously_known = self._state.get_active_paths(root)
            removed_paths = previously_known - seen_paths
            result.to_mark_deleted.extend(sorted(removed_paths))

        logger.info(f"Diff computed: {result.summary}")

        return result


def compute_diff(
    scanned_files: List[ScannedFile],
    vector_db_reader: VectorDBReader,
    state_reader: StateReader,
    collection: str,
    force: bool = False,
) -> DiffResult:
    """
    Convenience function to compute diff.

    Args:
        scanned_files: Files from the scanner
        vector_db_reader: Reader for checking vector DB existence
        state_reader: Reader for state (deletion detection only)
        collection: Vector DB collection name
        force: If True, ingest all files regardless of vector DB state

    Returns:
        DiffResult with action plan
    """
    differ = Differ(vector_db_reader, state_reader, collection)
    return differ.compute_diff(scanned_files, force=force)


__all__ = [
    "VectorDBReader",
    "StateReader",
    "FileCandidate",
    "DiffResult",
    "Differ",
    "compute_diff",
]