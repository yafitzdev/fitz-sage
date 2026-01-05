# fitz_ai/ingestion/source/plugins/filesystem.py
"""
Local filesystem source for file discovery.

Discovers files from local directories, supporting glob patterns
and recursive traversal.
"""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set

from fitz_ai.ingestion.source.base import SourceFile

logger = logging.getLogger(__name__)

# Default file extensions to discover
DEFAULT_EXTENSIONS: Set[str] = {
    # Documents
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    # Code
    ".py",
    ".js",
    ".ts",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    # Images (for OCR)
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
}


@dataclass
class FileSystemSource:
    """
    Source for discovering files from local filesystem.

    Example:
        source = FileSystemSource()
        for file in source.discover("./docs", patterns=["*.pdf", "*.md"]):
            print(file.local_path)
    """

    plugin_name: str = field(default="filesystem", repr=False)
    default_extensions: Set[str] = field(default_factory=lambda: DEFAULT_EXTENSIONS)
    recursive: bool = True
    follow_symlinks: bool = False

    def discover(
        self,
        root: str,
        patterns: Optional[List[str]] = None,
    ) -> Iterable[SourceFile]:
        """
        Discover files from a local directory or file path.

        Args:
            root: Path to file or directory.
            patterns: Optional glob patterns (e.g., ["*.pdf", "*.md"]).
                      If None, uses default_extensions.

        Yields:
            SourceFile for each discovered file.
        """
        root_path = Path(root).resolve()

        if not root_path.exists():
            logger.warning(f"Path does not exist: {root_path}")
            return

        if root_path.is_file():
            # Single file
            yield self._create_source_file(root_path)
            return

        # Directory - discover files
        if patterns:
            # Use provided patterns
            for pattern in patterns:
                glob_method = root_path.rglob if self.recursive else root_path.glob
                for path in glob_method(pattern):
                    if path.is_file():
                        yield self._create_source_file(path)
        else:
            # Use default extensions
            glob_method = root_path.rglob if self.recursive else root_path.glob
            for path in glob_method("*"):
                if path.is_file() and path.suffix.lower() in self.default_extensions:
                    yield self._create_source_file(path)

    def _create_source_file(self, path: Path) -> SourceFile:
        """Create a SourceFile from a local path."""
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(path))

        # Get file size
        try:
            size = path.stat().st_size
        except OSError:
            size = None

        return SourceFile(
            uri=path.as_uri(),  # file:///path/to/file
            local_path=path,
            mime_type=mime_type,
            size=size,
            metadata={
                "filename": path.name,
                "extension": path.suffix.lower(),
            },
        )


__all__ = ["FileSystemSource", "DEFAULT_EXTENSIONS"]
