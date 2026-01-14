# tests/test_vector_search_step_faiss_integration.py
"""
Integration test for VectorSearchStep with FaissLocalVectorDB.

This test ensures that VectorSearchStep can call FaissLocalVectorDB.search()
with all required parameters including query_filter.
"""

from pathlib import Path

import pytest

pytest.importorskip("faiss")

from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB
from fitz_ai.engines.fitz_rag.retrieval.steps.vector_search import VectorSearchStep


class FakeEmbedder:
    """Fake embedder that returns a fixed vector."""

    def __init__(self, vector: list[float]):
        self._vector = vector

    def embed(self, text: str) -> list[float]:
        return self._vector


def test_vector_search_step_with_faiss(tmp_path: Path):
    """Test VectorSearchStep works with FaissLocalVectorDB."""
    db = FaissLocalVectorDB(path=tmp_path)

    # Insert test data
    db.upsert(
        "test_collection",
        [
            {"id": "1", "vector": [1.0, 0.0, 0.0], "payload": {"content": "doc one"}},
            {"id": "2", "vector": [0.9, 0.1, 0.0], "payload": {"content": "doc two"}},
            {"id": "3", "vector": [0.0, 1.0, 0.0], "payload": {"content": "doc three"}},
        ],
    )

    # Create step with FAISS as client
    step = VectorSearchStep(
        client=db,
        embedder=FakeEmbedder([1.0, 0.0, 0.0]),
        collection="test_collection",
        k=10,
    )

    # Execute should work without TypeError
    chunks = step.execute("test query", [])

    assert len(chunks) == 3
    assert chunks[0].content == "doc one"  # Closest to query vector


def test_vector_search_step_with_faiss_and_filter(tmp_path: Path):
    """Test VectorSearchStep works with FaissLocalVectorDB and filter_conditions."""
    db = FaissLocalVectorDB(path=tmp_path)

    # Insert test data with metadata for filtering
    db.upsert(
        "test_collection",
        [
            {
                "id": "1",
                "vector": [1.0, 0.0, 0.0],
                "payload": {"content": "doc one", "category": "A"},
            },
            {
                "id": "2",
                "vector": [0.9, 0.1, 0.0],
                "payload": {"content": "doc two", "category": "B"},
            },
            {
                "id": "3",
                "vector": [0.8, 0.2, 0.0],
                "payload": {"content": "doc three", "category": "A"},
            },
        ],
    )

    # Create step with filter_conditions
    step = VectorSearchStep(
        client=db,
        embedder=FakeEmbedder([1.0, 0.0, 0.0]),
        collection="test_collection",
        k=10,
        filter_conditions={"key": "category", "match": {"value": "A"}},
    )

    # Execute should apply filter
    chunks = step.execute("test query", [])

    assert len(chunks) == 2
    assert all(c.metadata.get("category") == "A" for c in chunks)
