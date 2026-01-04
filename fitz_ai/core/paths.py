# fitz_ai/core/paths.py
"""
Central path management for Fitz.

ALL components that need file paths should use this module.
No hardcoded paths anywhere else in the codebase.

Design principles:
- Single source of truth
- Workspace-relative by default (CWD/.fitz/)
- Easy to override for testing
- No magic, explicit paths
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class FitzPaths:
    """
    Central path management for Fitz.

    All paths are relative to the workspace root, which defaults to
    the current working directory. This can be overridden for testing
    or special deployment scenarios.

    Usage:
        from fitz_ai.core.paths import FitzPaths

        # Get paths
        config_path = FitzPaths.config()
        vector_db_path = FitzPaths.vector_db()
        ingest_state_path = FitzPaths.ingest_state()

        # Override workspace for testing
        FitzPaths.set_workspace("/tmp/test_fitz")
    """

    _workspace_override: Optional[Path] = None

    @classmethod
    def set_workspace(cls, path: Optional[str | Path]) -> None:
        """
        Override the workspace root.

        Useful for testing or running multiple isolated instances.
        Pass None to reset to default (CWD).

        Args:
            path: New workspace root, or None to reset
        """
        if path is None:
            cls._workspace_override = None
        else:
            cls._workspace_override = Path(path)

    @classmethod
    def reset(cls) -> None:
        """Reset to default workspace (CWD). Useful in tests."""
        cls._workspace_override = None

    # =========================================================================
    # Core Paths
    # =========================================================================

    @classmethod
    def workspace(cls) -> Path:
        """
        The .fitz workspace directory.

        Default: {CWD}/.fitz/

        This is the root for all Fitz data in a project.
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

    # =========================================================================
    # Configuration
    # =========================================================================

    @classmethod
    def config(cls) -> Path:
        """
        Default config file path.

        Location: {workspace}/config.yaml
        """
        return cls.workspace() / "config.yaml"

    @classmethod
    def config_dir(cls) -> Path:
        """
        Config directory for engine-specific config files.

        Location: {workspace}/config/
        """
        return cls.workspace() / "config"

    @classmethod
    def engine_config(cls, engine_name: str) -> Path:
        """
        Engine-specific config file path.

        Location: {workspace}/config/{engine_name}.yaml

        Args:
            engine_name: Name of the engine (fitz_rag, graphrag, clara)
        """
        return cls.config_dir() / f"{engine_name}.yaml"

    @classmethod
    def ensure_config_dir(cls) -> Path:
        """Get config directory and create it if it doesn't exist."""
        path = cls.config_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =========================================================================
    # Ingestion State
    # =========================================================================

    @classmethod
    def ingest_state(cls) -> Path:
        """
        Ingestion state file for incremental ingestion.

        Location: {workspace}/ingest.json

        This file tracks:
        - Which files have been ingested
        - Content hashes for change detection
        - Deletion tracking
        - Config snapshots for staleness detection
        """
        return cls.workspace() / "ingest.json"

    # =========================================================================
    # Vector Database
    # =========================================================================

    @classmethod
    def vector_db(cls, collection: Optional[str] = None) -> Path:
        """
        Local vector database storage path.

        Location: {workspace}/vector_db/
        Or with collection: {workspace}/vector_db/{collection}/

        Used by local vector DB implementations (FAISS, etc.)
        """
        base = cls.workspace() / "vector_db"
        if collection:
            return base / collection
        return base

    @classmethod
    def ensure_vector_db(cls, collection: Optional[str] = None) -> Path:
        """Get vector DB path and create it if it doesn't exist."""
        path = cls.vector_db(collection)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =========================================================================
    # GraphRAG Storage
    # =========================================================================

    @classmethod
    def graphrag_storage(cls, collection: str) -> Path:
        """
        GraphRAG knowledge graph storage path.

        Location: {workspace}/graphrag/{collection}.json
        """
        return cls.workspace() / "graphrag" / f"{collection}.json"

    @classmethod
    def ensure_graphrag_storage(cls) -> Path:
        """Get graphrag directory and create it if it doesn't exist."""
        path = cls.workspace() / "graphrag"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =========================================================================
    # CLaRA Storage
    # =========================================================================

    @classmethod
    def clara_storage(cls, collection: str) -> Path:
        """
        CLaRA compressed representations storage path.

        Location: {workspace}/clara/{collection}/
        """
        return cls.workspace() / "clara" / collection

    @classmethod
    def ensure_clara_storage(cls, collection: str) -> Path:
        """Get clara collection directory and create it if it doesn't exist."""
        path = cls.clara_storage(collection)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =========================================================================
    # Cache Paths
    # =========================================================================

    @classmethod
    def cache(cls) -> Path:
        """
        Cache directory root.

        Location: {workspace}/cache/
        """
        return cls.workspace() / "cache"

    @classmethod
    def embeddings_cache(cls) -> Path:
        """
        Cached embeddings.

        Location: {workspace}/cache/embeddings/
        """
        return cls.workspace() / "cache" / "embeddings"

    @classmethod
    def chunks_cache(cls) -> Path:
        """
        Cached chunks.

        Location: {workspace}/cache/chunks/
        """
        return cls.workspace() / "cache" / "chunks"

    # =========================================================================
    # Knowledge Map
    # =========================================================================

    @classmethod
    def knowledge_map(cls) -> Path:
        """
        Knowledge map state file for visualization caching.

        Location: {workspace}/knowledge_map.json

        This file caches:
        - Chunk embeddings (float16) for incremental updates
        - Document metadata for hierarchy display
        - Collection and embedding config for cache invalidation
        """
        return cls.workspace() / "knowledge_map.json"

    @classmethod
    def knowledge_map_html(cls) -> Path:
        """
        Default output path for knowledge map HTML visualization.

        Location: {workspace}/knowledge_map.html
        """
        return cls.workspace() / "knowledge_map.html"

    # =========================================================================
    # Logs & Diagnostics
    # =========================================================================

    @classmethod
    def logs(cls) -> Path:
        """
        Log files directory.

        Location: {workspace}/logs/
        """
        return cls.workspace() / "logs"

    @classmethod
    def ensure_logs(cls) -> Path:
        """Get logs path and create it if it doesn't exist."""
        path = cls.logs()
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =========================================================================
    # Quickstart / Demo
    # =========================================================================

    @classmethod
    def quickstart_docs(cls) -> Path:
        """
        Sample documents created by quickstart.

        Location: {workspace}/quickstart_docs/
        """
        return cls.workspace() / "quickstart_docs"

    @classmethod
    def quickstart_config(cls) -> Path:
        """
        Config file created by quickstart.

        Location: {workspace}/quickstart_config.yaml
        """
        return cls.workspace() / "quickstart_config.yaml"

    # =========================================================================
    # User Plugins (Home Directory)
    # =========================================================================

    @classmethod
    def user_home(cls) -> Path:
        """
        User's fitz home directory.

        Location: ~/.fitz/

        This is separate from the workspace (which is project-specific).
        Used for user-created plugins and global configuration.
        """
        return Path.home() / ".fitz"

    @classmethod
    def user_plugins(cls) -> Path:
        """
        User plugins directory.

        Location: ~/.fitz/plugins/

        Structure:
            plugins/
            ├── llm/
            │   ├── chat/
            │   ├── embedding/
            │   └── rerank/
            ├── vector_db/
            ├── chunking/
            ├── reader/
            └── constraint/
        """
        return cls.user_home() / "plugins"

    @classmethod
    def user_llm_plugins(cls, plugin_type: str) -> Path:
        """
        User LLM plugins directory for a specific type.

        Location: ~/.fitz/plugins/llm/{plugin_type}/

        Args:
            plugin_type: One of 'chat', 'embedding', 'rerank'
        """
        return cls.user_plugins() / "llm" / plugin_type

    @classmethod
    def user_vector_db_plugins(cls) -> Path:
        """
        User vector DB plugins directory.

        Location: ~/.fitz/plugins/vector_db/
        """
        return cls.user_plugins() / "vector_db"

    @classmethod
    def user_chunking_plugins(cls) -> Path:
        """
        User chunking plugins directory.

        Location: ~/.fitz/plugins/chunking/
        """
        return cls.user_plugins() / "chunking"

    @classmethod
    def user_reader_plugins(cls) -> Path:
        """
        User reader plugins directory.

        Location: ~/.fitz/plugins/reader/
        """
        return cls.user_plugins() / "reader"

    @classmethod
    def user_constraint_plugins(cls) -> Path:
        """
        User constraint plugins directory.

        Location: ~/.fitz/plugins/constraint/
        """
        return cls.user_plugins() / "constraint"

    @classmethod
    def ensure_user_plugins(cls) -> Path:
        """Create user plugins directory structure if it doesn't exist."""
        base = cls.user_plugins()
        # Create all plugin subdirectories
        (base / "llm" / "chat").mkdir(parents=True, exist_ok=True)
        (base / "llm" / "embedding").mkdir(parents=True, exist_ok=True)
        (base / "llm" / "rerank").mkdir(parents=True, exist_ok=True)
        (base / "vector_db").mkdir(parents=True, exist_ok=True)
        (base / "chunking").mkdir(parents=True, exist_ok=True)
        (base / "reader").mkdir(parents=True, exist_ok=True)
        (base / "constraint").mkdir(parents=True, exist_ok=True)
        return base


# =============================================================================
# Convenience Functions
# =============================================================================


def get_workspace() -> Path:
    """Convenience function to get workspace path."""
    return FitzPaths.workspace()


def get_vector_db_path(collection: Optional[str] = None) -> Path:
    """Convenience function to get vector DB path."""
    return FitzPaths.vector_db(collection)


def get_config_path() -> Path:
    """Convenience function to get config path."""
    return FitzPaths.config()


def get_ingest_state_path() -> Path:
    """Convenience function to get ingest state path."""
    return FitzPaths.ingest_state()
