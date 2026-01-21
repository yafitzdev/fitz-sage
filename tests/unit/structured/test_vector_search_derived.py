# tests/unit/structured/test_vector_search_derived.py
"""
Tests for VectorSearchStep integration with derived collection.
"""

from dataclasses import dataclass
from typing import Any

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.retrieval.steps.vector_search import (
    DERIVED_AVAILABLE,
    VectorSearchStep,
)


@dataclass
class MockHit:
    """Mock vector search hit."""

    id: str
    score: float
    payload: dict[str, Any]


class MockEmbedder:
    """Mock embedder that returns fixed vectors."""

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class MockVectorClient:
    """Mock vector client for testing."""

    def __init__(
        self,
        main_results: list[MockHit] | None = None,
        derived_results: list[MockHit] | None = None,
    ):
        self._main_results = main_results or []
        self._derived_results = derived_results or []
        self._searched_collections: list[str] = []

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
        query_filter: dict | None = None,
    ) -> list[MockHit]:
        """Return mock results based on collection name."""
        self._searched_collections.append(collection_name)
        if collection_name.endswith("__derived"):
            return self._derived_results
        return self._main_results

    def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
    ) -> list[dict]:
        """Mock retrieve - return empty."""
        return []


class TestDerivedSearchIntegration:
    """Tests for derived collection search in VectorSearchStep."""

    def test_derived_available_flag(self):
        """Test that DERIVED_AVAILABLE flag is set."""
        assert DERIVED_AVAILABLE is True

    def test_search_derived_returns_empty_when_disabled(self):
        """Test that derived search returns empty when include_derived=False."""
        step = VectorSearchStep(
            client=MockVectorClient(),
            embedder=MockEmbedder(),
            collection="test_collection",
            include_derived=False,
        )

        result = step._search_derived([0.1, 0.2, 0.3])
        assert result == []

    def test_search_derived_returns_chunks_when_enabled(self):
        """Test that derived search returns chunks when enabled."""
        derived_hits = [
            MockHit(
                id="derived_1",
                score=0.95,
                payload={
                    "__derived": True,
                    "__source_table": "employees",
                    "content": "There are 5 employees in the database.",
                },
            ),
        ]

        step = VectorSearchStep(
            client=MockVectorClient(derived_results=derived_hits),
            embedder=MockEmbedder(),
            collection="test_collection",
            include_derived=True,
        )

        result = step._search_derived([0.1, 0.2, 0.3])

        assert len(result) == 1
        assert result[0].content == "There are 5 employees in the database."
        assert result[0].metadata.get("is_derived") is True
        assert result[0].metadata.get("source_table") == "employees"

    def test_search_derived_searches_correct_collection(self):
        """Test that derived search uses correct collection name."""
        client = MockVectorClient()

        step = VectorSearchStep(
            client=client,
            embedder=MockEmbedder(),
            collection="my_docs",
            include_derived=True,
        )

        step._search_derived([0.1, 0.2, 0.3])

        assert "my_docs__derived" in client._searched_collections

    def test_merge_derived_results_preserves_priority(self):
        """Test that derived results are placed at front of merged results."""
        step = VectorSearchStep(
            client=MockVectorClient(),
            embedder=MockEmbedder(),
            collection="test",
        )

        main_chunks = [
            Chunk(id="main_1", doc_id="doc1", content="Main content 1", chunk_index=0),
            Chunk(id="main_2", doc_id="doc1", content="Main content 2", chunk_index=1),
        ]

        derived_chunks = [
            Chunk(
                id="derived_1",
                doc_id="table:employees",
                content="Derived content",
                chunk_index=0,
                metadata={"is_derived": True},
            ),
        ]

        merged = step._merge_derived_results(main_chunks, derived_chunks)

        assert len(merged) == 3
        assert merged[0].id == "derived_1"
        assert merged[0].metadata.get("is_derived") is True
        assert merged[1].id == "main_1"
        assert merged[2].id == "main_2"

    def test_merge_derived_results_deduplicates(self):
        """Test that merge deduplicates by content."""
        step = VectorSearchStep(
            client=MockVectorClient(),
            embedder=MockEmbedder(),
            collection="test",
        )

        main_chunks = [
            Chunk(id="main_1", doc_id="doc1", content="Same content", chunk_index=0),
        ]

        derived_chunks = [
            Chunk(
                id="derived_1",
                doc_id="table:t",
                content="Same content",  # Same content as main
                chunk_index=0,
            ),
        ]

        merged = step._merge_derived_results(main_chunks, derived_chunks)

        # Should have only 1 result (derived takes priority)
        assert len(merged) == 1
        assert merged[0].id == "derived_1"

    def test_merge_derived_empty_derived_returns_main(self):
        """Test that empty derived list returns main unchanged."""
        step = VectorSearchStep(
            client=MockVectorClient(),
            embedder=MockEmbedder(),
            collection="test",
        )

        main_chunks = [
            Chunk(id="main_1", doc_id="doc1", content="Content", chunk_index=0),
        ]

        merged = step._merge_derived_results(main_chunks, [])

        assert merged is main_chunks
        assert len(merged) == 1

    def test_single_search_includes_derived(self):
        """Test that single search includes derived results."""
        main_hits = [
            MockHit(
                id="main_1",
                score=0.8,
                payload={"content": "Main document content", "doc_id": "doc1"},
            ),
        ]

        derived_hits = [
            MockHit(
                id="derived_1",
                score=0.95,
                payload={
                    "__derived": True,
                    "__source_table": "employees",
                    "content": "There are 5 employees.",
                },
            ),
        ]

        client = MockVectorClient(main_results=main_hits, derived_results=derived_hits)

        step = VectorSearchStep(
            client=client,
            embedder=MockEmbedder(),
            collection="test",
            include_derived=True,
            k=10,
        )

        result = step._single_search("how many employees?", [])

        # Should have searched both collections
        assert "test" in client._searched_collections
        assert "test__derived" in client._searched_collections

        # Should have both types of results
        derived = [c for c in result if c.metadata.get("is_derived")]
        main = [c for c in result if not c.metadata.get("is_derived")]

        assert len(derived) > 0
        assert len(main) > 0


class TestDerivedChunkMetadata:
    """Tests for derived chunk metadata."""

    def test_derived_chunk_has_correct_doc_id(self):
        """Test that derived chunks have table-prefixed doc_id."""
        derived_hits = [
            MockHit(
                id="d1",
                score=0.9,
                payload={
                    "__source_table": "my_table",
                    "content": "Content",
                },
            ),
        ]

        step = VectorSearchStep(
            client=MockVectorClient(derived_results=derived_hits),
            embedder=MockEmbedder(),
            collection="test",
            include_derived=True,
        )

        result = step._search_derived([0.1, 0.2, 0.3])

        assert result[0].doc_id == "table:my_table"

    def test_derived_chunk_id_prefixed(self):
        """Test that derived chunk IDs are prefixed."""
        derived_hits = [
            MockHit(
                id="abc123",
                score=0.9,
                payload={"content": "Content"},
            ),
        ]

        step = VectorSearchStep(
            client=MockVectorClient(derived_results=derived_hits),
            embedder=MockEmbedder(),
            collection="test",
            include_derived=True,
        )

        result = step._search_derived([0.1, 0.2, 0.3])

        assert result[0].id == "derived_abc123"
