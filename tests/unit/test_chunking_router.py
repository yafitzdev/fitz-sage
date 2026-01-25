# tests/test_chunking_router.py
"""
Tests for ChunkingRouter.

Verifies:
1. Router correctly routes documents to file-type specific chunkers
2. Fallback to default chunker works with warning
3. Configuration parsing and normalization works
"""

import logging
from dataclasses import dataclass
from typing import List

import pytest

from fitz_ai.core.chunk import Chunk
from fitz_ai.core.document import ParsedDocument
from fitz_ai.engines.fitz_rag.config import (
    ChunkingRouterConfig,
    ExtensionChunkerConfig,
)
from fitz_ai.ingestion.chunking.plugins.default.simple import SimpleChunker
from fitz_ai.ingestion.chunking.router import ChunkingRouter

# =============================================================================
# Mock Chunkers for Testing
# =============================================================================


@dataclass
class MockMarkdownChunker:
    """Mock markdown chunker for testing."""

    plugin_name: str = "markdown"
    max_tokens: int = 800

    @property
    def chunker_id(self) -> str:
        return f"{self.plugin_name}:{self.max_tokens}"

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        text = document.full_text
        doc_id = document.metadata.get("doc_id", "test")
        return [
            Chunk(
                id="md:0",
                doc_id=doc_id,
                chunk_index=0,
                content=f"[MARKDOWN] {text[:50]}",
                metadata=document.metadata,
            )
        ]


@dataclass
class MockPythonChunker:
    """Mock Python code chunker for testing."""

    plugin_name: str = "python_code"
    chunk_by: str = "function"

    @property
    def chunker_id(self) -> str:
        return f"{self.plugin_name}:{self.chunk_by}"

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        text = document.full_text
        doc_id = document.metadata.get("doc_id", "test")
        return [
            Chunk(
                id="py:0",
                doc_id=doc_id,
                chunk_index=0,
                content=f"[PYTHON] {text[:50]}",
                metadata=document.metadata,
            )
        ]


# =============================================================================
# Router Construction Tests
# =============================================================================


class TestRouterConstruction:
    """Tests for ChunkingRouter construction."""

    def test_basic_construction(self):
        """Router can be constructed with chunker map and default."""
        default = SimpleChunker(chunk_size=1000)
        router = ChunkingRouter(
            chunker_map={},
            default_chunker=default,
        )

        assert router.default_chunker is default
        assert router.configured_extensions == []

    def test_construction_with_extensions(self):
        """Router tracks configured extensions."""
        default = SimpleChunker()
        md_chunker = MockMarkdownChunker()
        py_chunker = MockPythonChunker()

        router = ChunkingRouter(
            chunker_map={".md": md_chunker, ".py": py_chunker},
            default_chunker=default,
        )

        assert ".md" in router.configured_extensions
        assert ".py" in router.configured_extensions
        assert len(router.configured_extensions) == 2


# =============================================================================
# Routing Tests
# =============================================================================


class TestRouting:
    """Tests for document routing logic."""

    @pytest.fixture
    def router(self):
        """Create a router with md and py chunkers."""
        return ChunkingRouter(
            chunker_map={
                ".md": MockMarkdownChunker(),
                ".py": MockPythonChunker(),
            },
            default_chunker=SimpleChunker(chunk_size=1000),
            warn_on_fallback=True,
        )

    def test_routes_md_to_markdown_chunker(self, router):
        """Markdown files route to markdown chunker."""
        chunker = router.get_chunker(".md")
        assert chunker.plugin_name == "markdown"

    def test_routes_py_to_python_chunker(self, router):
        """Python files route to python chunker."""
        chunker = router.get_chunker(".py")
        assert chunker.plugin_name == "python_code"

    def test_unknown_extension_falls_back_to_default(self, router):
        """Unknown extensions fall back to default chunker."""
        chunker = router.get_chunker(".xyz")
        assert chunker.plugin_name == "simple"

    def test_txt_falls_back_to_default(self, router):
        """Unconfigured .txt falls back to default."""
        chunker = router.get_chunker(".txt")
        assert chunker.plugin_name == "simple"

    def test_extension_normalization(self, router):
        """Extensions are normalized to lowercase with dot."""
        assert router.get_chunker("md").plugin_name == "markdown"
        assert router.get_chunker(".MD").plugin_name == "markdown"
        assert router.get_chunker("Py").plugin_name == "python_code"

    def test_get_chunker_id(self, router):
        """get_chunker_id returns correct ID for extension."""
        assert router.get_chunker_id(".md") == "markdown:800"
        assert router.get_chunker_id(".py") == "python_code:function"
        assert router.get_chunker_id(".txt") == "simple:1000:0"


class TestFallbackWarning:
    """Tests for fallback warning behavior."""

    def test_warns_on_unknown_extension(self, caplog):
        """Logs warning when falling back to default."""
        router = ChunkingRouter(
            chunker_map={".md": MockMarkdownChunker()},
            default_chunker=SimpleChunker(),
            warn_on_fallback=True,
        )

        with caplog.at_level(logging.WARNING):
            router.get_chunker(".unknown")

        assert "No chunker configured for extension '.unknown'" in caplog.text
        assert "using default chunker" in caplog.text

    def test_warns_only_once_per_extension(self, caplog):
        """Only warns once per unknown extension."""
        router = ChunkingRouter(
            chunker_map={},
            default_chunker=SimpleChunker(),
            warn_on_fallback=True,
        )

        with caplog.at_level(logging.WARNING):
            router.get_chunker(".xyz")
            router.get_chunker(".xyz")
            router.get_chunker(".xyz")

        assert caplog.text.count("No chunker configured") == 1

    def test_no_warning_when_disabled(self, caplog):
        """No warning when warn_on_fallback is False."""
        router = ChunkingRouter(
            chunker_map={},
            default_chunker=SimpleChunker(),
            warn_on_fallback=False,
        )

        with caplog.at_level(logging.WARNING):
            router.get_chunker(".unknown")

        assert "No chunker configured" not in caplog.text


# =============================================================================
# Configuration Tests
# =============================================================================


class TestRouterConfig:
    """Tests for ChunkingRouterConfig."""

    def test_extension_normalization_in_config(self):
        """Extensions in config are normalized."""
        config = ChunkingRouterConfig(
            default=ExtensionChunkerConfig(plugin_name="simple"),
            by_extension={
                "md": ExtensionChunkerConfig(plugin_name="markdown"),
                ".PY": ExtensionChunkerConfig(plugin_name="python"),
                ".Txt": ExtensionChunkerConfig(plugin_name="text"),
            },
        )

        assert ".md" in config.by_extension
        assert ".py" in config.by_extension
        assert ".txt" in config.by_extension


# =============================================================================
# Document Chunking Tests
# =============================================================================


class TestDocumentChunking:
    """Tests for chunking ParsedDocuments."""

    def test_chunker_chunks_parsed_document(self):
        """Chunker correctly chunks a ParsedDocument."""
        from fitz_ai.core.document import DocumentElement, ElementType

        router = ChunkingRouter(
            chunker_map={".md": MockMarkdownChunker()},
            default_chunker=SimpleChunker(chunk_size=100),
        )

        doc = ParsedDocument(
            source="file:///docs/readme.md",
            elements=[
                DocumentElement(
                    type=ElementType.TEXT,
                    content="# Hello World\n\nThis is a test.",
                )
            ],
            metadata={"doc_id": "readme"},
        )

        chunker = router.get_chunker(".md")
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert "[MARKDOWN]" in chunks[0].content

    def test_default_chunker_for_unknown_extension(self):
        """Default chunker is used for unknown extensions."""
        from fitz_ai.core.document import DocumentElement, ElementType
        from fitz_ai.ingestion.chunking.plugins.default.recursive import RecursiveChunker

        router = ChunkingRouter(
            chunker_map={},
            default_chunker=RecursiveChunker(chunk_size=100, chunk_overlap=20),
            warn_on_fallback=False,
        )

        doc = ParsedDocument(
            source="file:///docs/data.xyz",
            elements=[
                DocumentElement(
                    type=ElementType.TEXT,
                    content="This is some text content for an unknown file extension that should use the default chunker instead of any specialized one.",
                )
            ],
            metadata={"doc_id": "data"},
        )

        chunker = router.get_chunker(".xyz")
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        assert "[MARKDOWN]" not in chunks[0].content
