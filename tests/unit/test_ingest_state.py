# tests/unit/test_ingest_state.py
"""
Tests for fitz_ai.ingestion.state module.
"""

import pytest

from tests.conftest import POSTGRES_DEPS_AVAILABLE, SKIP_POSTGRES_REASON

# Skip entire module if postgres dependencies not available
# (IngestStateManager uses PostgreSQL storage)
if not POSTGRES_DEPS_AVAILABLE:
    pytest.skip(SKIP_POSTGRES_REASON, allow_module_level=True)

from fitz_ai.ingestion.state import (
    EmbeddingConfig,
    FileEntry,
    FileStatus,
    IngestState,
    IngestStateManager,
)


class TestIngestState:
    """Tests for IngestState model (Pydantic schema)."""

    def test_create_default_state(self):
        """Test creating a default state."""
        state = IngestState(project_id="test-123")

        assert state.schema_version == 1
        assert state.project_id == "test-123"
        assert state.roots == {}

    def test_get_active_paths(self):
        """Test getting active paths from state."""
        state = IngestState(project_id="test")

        # No paths initially
        assert state.get_active_paths("/root") == set()

    def test_get_file_entry(self):
        """Test getting file entry."""
        state = IngestState(project_id="test")

        # No entry initially
        assert state.get_file_entry("/root", "/root/file.md") is None


class TestFileEntry:
    """Tests for FileEntry model."""

    def test_is_active(self):
        """Test is_active method."""
        active = FileEntry(
            ext=".md",
            size_bytes=100,
            mtime_epoch=1234567890.0,
            content_hash="sha256:abc",
            status=FileStatus.ACTIVE,
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )
        assert active.is_active() is True

        deleted = FileEntry(
            ext=".md",
            size_bytes=100,
            mtime_epoch=1234567890.0,
            content_hash="sha256:abc",
            status=FileStatus.DELETED,
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )
        assert deleted.is_active() is False


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig model."""

    def test_create(self):
        """Test creating embedding config."""
        config = EmbeddingConfig.create("openai", "text-embedding-3-small")

        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.id == "openai:text-embedding-3-small"

    def test_create_with_extras(self):
        """Test creating embedding config with dimension."""
        config = EmbeddingConfig.create(
            "openai",
            "text-embedding-3-small",
            dimension=1536,
        )

        assert config.dimension == 1536


class TestIngestStateManager:
    """Tests for IngestStateManager (PostgreSQL-backed)."""

    def test_creates_new_state(self):
        """Test that new state is created if not exists."""
        manager = IngestStateManager()
        manager.load()

        assert manager.schema_version == 1
        assert len(manager.project_id) > 0

    def test_mark_active(self):
        """Test marking a file as active."""
        manager = IngestStateManager()
        manager.load()

        manager.mark_active(
            file_path="/root/test_state_active.md",
            root="/root",
            content_hash="sha256:abc123",
            ext=".md",
            size_bytes=1234,
            mtime_epoch=1234567890.0,
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )

        entry = manager.get_file_entry("/root", "/root/test_state_active.md")
        assert entry is not None
        assert entry.content_hash == "sha256:abc123"
        assert entry.status == FileStatus.ACTIVE
        assert entry.chunker_id == "simple:1000:0"
        assert entry.parser_id == "md.v1"
        assert entry.embedding_id == "cohere:embed-english-v3.0"

    def test_mark_deleted(self):
        """Test marking a file as deleted."""
        manager = IngestStateManager()
        manager.load()

        # First mark as active
        manager.mark_active(
            file_path="/root/test_state_deleted.md",
            root="/root",
            content_hash="sha256:abc123",
            ext=".md",
            size_bytes=1234,
            mtime_epoch=1234567890.0,
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )

        # Then mark as deleted
        manager.mark_deleted("/root", "/root/test_state_deleted.md")

        entry = manager.get_file_entry("/root", "/root/test_state_deleted.md")
        assert entry is not None
        assert entry.status == FileStatus.DELETED

    def test_persistence_across_managers(self):
        """Test that state persists across manager instances."""
        manager = IngestStateManager()
        manager.load()

        manager.mark_active(
            file_path="/root/test_state_persist.md",
            root="/root",
            content_hash="sha256:abc123",
            ext=".md",
            size_bytes=1234,
            mtime_epoch=1234567890.0,
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )
        manager.save()

        # Create new manager and verify data persists
        manager2 = IngestStateManager()
        manager2.load()

        entry = manager2.get_file_entry("/root", "/root/test_state_persist.md")
        assert entry is not None
        assert entry.content_hash == "sha256:abc123"
        assert entry.chunker_id == "simple:1000:0"

    def test_get_active_paths(self):
        """Test getting active paths."""
        manager = IngestStateManager()
        manager.load()

        # Add some files with unique paths
        manager.mark_active(
            "/test_active_root/a.md",
            "/test_active_root",
            "sha256:a",
            ".md",
            100,
            1234567890.0,
            "simple:1000:0",
            "md.v1",
            "cohere:embed-english-v3.0",
        )
        manager.mark_active(
            "/test_active_root/b.md",
            "/test_active_root",
            "sha256:b",
            ".md",
            100,
            1234567890.0,
            "simple:1000:0",
            "md.v1",
            "cohere:embed-english-v3.0",
        )
        manager.mark_active(
            "/test_active_root/c.md",
            "/test_active_root",
            "sha256:c",
            ".md",
            100,
            1234567890.0,
            "simple:1000:0",
            "md.v1",
            "cohere:embed-english-v3.0",
        )

        # Delete one
        manager.mark_deleted("/test_active_root", "/test_active_root/b.md")

        active = manager.get_active_paths("/test_active_root")
        assert "/test_active_root/a.md" in active
        assert "/test_active_root/c.md" in active
        assert "/test_active_root/b.md" not in active

    def test_set_embedding_config(self):
        """Test setting embedding config."""
        manager = IngestStateManager()
        manager.load()

        manager.set_embedding_config("openai", "text-embedding-3-small")

        assert manager.state.embedding is not None
        assert manager.state.embedding.id == "openai:text-embedding-3-small"

    def test_config_ids_persisted(self):
        """Test that config IDs are persisted and loaded correctly."""
        manager = IngestStateManager()
        manager.load()

        manager.mark_active(
            file_path="/root/test_config_persist.md",
            root="/root",
            content_hash="sha256:abc123",
            ext=".md",
            size_bytes=1234,
            mtime_epoch=1234567890.0,
            chunker_id="markdown:800:100",
            parser_id="md.v2",
            embedding_id="openai:text-embedding-3-large",
        )
        manager.save()

        # Reload and verify
        manager2 = IngestStateManager()
        manager2.load()

        entry = manager2.get_file_entry("/root", "/root/test_config_persist.md")
        assert entry.chunker_id == "markdown:800:100"
        assert entry.parser_id == "md.v2"
        assert entry.embedding_id == "openai:text-embedding-3-large"
