# fitz_ai/cli/commands/ingest_helpers.py
"""
Helper functions for ingest command.

Content detection, artifact discovery, and utility functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def suggest_collection_name(source: str) -> str:
    """Suggest a collection name from source path."""
    path = Path(source).resolve()
    # Use folder name, sanitized
    name = path.name if path.is_dir() else path.parent.name
    # Replace spaces/special chars with underscores
    return name.replace(" ", "_").replace("-", "_").lower()


def detect_content_type(source: str) -> tuple[str, str]:
    """
    Detect whether source is a codebase or document corpus.

    Returns:
        Tuple of (content_type, reason) where content_type is "codebase" or "documents"
    """
    from fitz_ai.ingestion.detection import detect_content_type as _detect

    result = _detect(Path(source))
    return result.content_type, result.reason


def is_code_project(source: str) -> bool:
    """Check if source is a codebase."""
    content_type, _ = detect_content_type(source)
    return content_type == "codebase"


def get_available_artifacts(has_llm: bool = False) -> List[tuple]:
    """
    Get available artifact plugins as (name, description) tuples.

    Args:
        has_llm: Whether an LLM client is available (enables LLM-requiring artifacts)

    Returns:
        List of (name, description) tuples for available artifacts
    """
    from fitz_ai.ingestion.enrichment.artifacts.registry import get_artifact_registry

    registry = get_artifact_registry()
    result = []

    for name in registry.list_plugin_names():
        info = registry.get_plugin(name)
        if info is None:
            continue

        # Skip LLM-requiring artifacts if no LLM available
        if info.requires_llm and not has_llm:
            desc = f"{info.description} (requires LLM)"
        else:
            desc = info.description

        result.append((name, desc))

    return result


def parse_artifact_selection(
    artifacts_arg: Optional[str], available: List[str]
) -> Optional[List[str]]:
    """
    Parse the --artifacts argument.

    Args:
        artifacts_arg: The --artifacts argument value
        available: List of available artifact names

    Returns:
        List of selected artifact names, or None if should prompt interactively
    """
    if artifacts_arg is None:
        return None  # Interactive selection

    artifacts_arg = artifacts_arg.strip().lower()

    if artifacts_arg == "all":
        return available
    elif artifacts_arg == "none":
        return []
    else:
        # Comma-separated list
        requested = [a.strip() for a in artifacts_arg.split(",")]
        # Filter to valid names
        return [a for a in requested if a in available]


def is_direct_text(source: str) -> bool:
    """
    Determine if source is direct text rather than a file path.

    Returns True if:
    - Source doesn't exist as a path
    - Source doesn't look like a path (no slashes, backslashes, or common extensions)
    - Source contains spaces or looks like natural language

    Returns:
        True if source should be treated as direct text
    """
    # Check if it exists as a file/directory
    path = Path(source)
    if path.exists():
        return False

    # Common file extensions that indicate a path (even if file doesn't exist)
    path_extensions = {
        ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".py", ".js", ".ts",
        ".html", ".css", ".xml", ".pdf", ".doc", ".docx", ".xls", ".xlsx"
    }

    # Check if it looks like a path
    source_lower = source.lower()

    # Has path separators
    if "/" in source or "\\" in source:
        return False

    # Has a file extension
    if any(source_lower.endswith(ext) for ext in path_extensions):
        return False

    # Looks like a relative path (starts with . or ~)
    if source.startswith(".") or source.startswith("~"):
        return False

    # If it has spaces and no path-like characters, it's likely text
    # Or if it's longer than typical filenames
    if " " in source or len(source) > 100:
        return True

    # Default to treating as path (let the path validation fail later)
    return False
