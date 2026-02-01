# tests/unit/property/test_chunkers.py
"""
Property-based tests for SimpleChunker and RecursiveChunker.

Tests pure, deterministic properties of chunking logic.
Targets:
    - fitz_ai/ingestion/chunking/plugins/default/simple.py
    - fitz_ai/ingestion/chunking/plugins/default/recursive.py
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from fitz_ai.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_ai.ingestion.chunking.plugins.default.recursive import RecursiveChunker
from fitz_ai.ingestion.chunking.plugins.default.simple import SimpleChunker

from .strategies import chunk_params, document_text

pytestmark = pytest.mark.property


def make_document(text: str, doc_id: str = "test_doc") -> ParsedDocument:
    """Helper to create a ParsedDocument from text."""
    return ParsedDocument(
        source=f"file:///{doc_id}.txt",
        elements=[DocumentElement(type=ElementType.TEXT, content=text)],
        metadata={"doc_id": doc_id},
    )


# =============================================================================
# SimpleChunker Tests
# =============================================================================


class TestSimpleChunkerNonEmpty:
    """Test that non-empty input produces chunks."""

    @given(text=document_text(min_chars=10, max_chars=2000))
    @settings(max_examples=50)
    def test_non_empty_input_produces_chunks(self, text: str):
        """Non-empty document produces at least one chunk."""
        assume(text.strip())

        chunker = SimpleChunker(chunk_size=500, chunk_overlap=0)
        doc = make_document(text)

        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1

    @given(params=chunk_params(min_size=50, max_size=1000))
    def test_produces_chunks_with_various_params(self, params: tuple[int, int]):
        """Various valid params produce chunks from non-empty input."""
        chunk_size, chunk_overlap = params

        chunker = SimpleChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc = make_document("A" * (chunk_size * 2))  # Enough text for multiple chunks

        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1


class TestSimpleChunkerSizeConstraint:
    """Test that chunk content never exceeds chunk_size."""

    @given(
        text=document_text(min_chars=100, max_chars=3000),
        params=chunk_params(min_size=50, max_size=500),
    )
    @settings(max_examples=50)
    def test_chunk_size_respected(self, text: str, params: tuple[int, int]):
        """Each chunk.content length <= chunk_size."""
        assume(text.strip())
        chunk_size, chunk_overlap = params

        chunker = SimpleChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc = make_document(text)

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert (
                len(chunk.content) <= chunk_size
            ), f"Chunk content {len(chunk.content)} > {chunk_size}"


class TestSimpleChunkerSequentialIndices:
    """Test that chunk_index is 0, 1, 2, ..."""

    @given(text=document_text(min_chars=100, max_chars=2000))
    @settings(max_examples=50)
    def test_sequential_chunk_indices(self, text: str):
        """chunk_index values are 0, 1, 2, ... in order."""
        assume(text.strip())

        chunker = SimpleChunker(chunk_size=200, chunk_overlap=50)
        doc = make_document(text)

        chunks = chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i, f"Expected index {i}, got {chunk.chunk_index}"


class TestSimpleChunkerSequentialIds:
    """Test that chunk.id follows doc_id:N pattern."""

    @given(text=document_text(min_chars=100, max_chars=2000))
    @settings(max_examples=50)
    def test_sequential_chunk_ids(self, text: str):
        """chunk.id follows 'doc_id:N' pattern."""
        assume(text.strip())

        doc_id = "my_test_doc"
        chunker = SimpleChunker(chunk_size=200, chunk_overlap=50)
        doc = make_document(text, doc_id=doc_id)

        chunks = chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            expected_id = f"{doc_id}:{i}"
            assert chunk.id == expected_id, f"Expected {expected_id}, got {chunk.id}"


class TestSimpleChunkerDeterminism:
    """Test that same input produces same output."""

    @given(text=document_text(min_chars=50, max_chars=1000))
    @settings(max_examples=30)
    def test_deterministic_output(self, text: str):
        """Same input produces identical chunks."""
        assume(text.strip())

        chunker = SimpleChunker(chunk_size=200, chunk_overlap=50)
        doc = make_document(text)

        chunks1 = chunker.chunk(doc)
        chunks2 = chunker.chunk(doc)

        assert len(chunks1) == len(chunks2)

        for c1, c2 in zip(chunks1, chunks2):
            assert c1.id == c2.id
            assert c1.content == c2.content
            assert c1.chunk_index == c2.chunk_index


class TestSimpleChunkerValidation:
    """Test that invalid params are rejected."""

    @given(size=st.integers(min_value=-1000, max_value=0))
    def test_rejects_invalid_chunk_size(self, size: int):
        """chunk_size <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size"):
            SimpleChunker(chunk_size=size, chunk_overlap=0)

    @given(overlap=st.integers(min_value=-1000, max_value=-1))
    def test_rejects_negative_overlap(self, overlap: int):
        """chunk_overlap < 0 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            SimpleChunker(chunk_size=100, chunk_overlap=overlap)

    @given(size=st.integers(min_value=10, max_value=100))
    def test_rejects_overlap_gte_size(self, size: int):
        """chunk_overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            SimpleChunker(chunk_size=size, chunk_overlap=size)

        with pytest.raises(ValueError, match="chunk_overlap"):
            SimpleChunker(chunk_size=size, chunk_overlap=size + 1)


class TestSimpleChunkerIdDeterminism:
    """Test that chunker_id is deterministic."""

    @given(params=chunk_params(min_size=10, max_size=1000))
    def test_chunker_id_deterministic(self, params: tuple[int, int]):
        """Same params produce same chunker_id."""
        chunk_size, chunk_overlap = params

        chunker1 = SimpleChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunker2 = SimpleChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        assert chunker1.chunker_id == chunker2.chunker_id

    @given(params=chunk_params(min_size=10, max_size=1000))
    def test_chunker_id_format(self, params: tuple[int, int]):
        """chunker_id follows 'simple:size:overlap' format."""
        chunk_size, chunk_overlap = params

        chunker = SimpleChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        expected = f"simple:{chunk_size}:{chunk_overlap}"
        assert chunker.chunker_id == expected


# =============================================================================
# RecursiveChunker Tests
# =============================================================================


class TestRecursiveChunkerNonEmpty:
    """Test that non-empty input produces chunks."""

    @given(text=document_text(min_chars=10, max_chars=2000))
    @settings(max_examples=50)
    def test_non_empty_input_produces_chunks(self, text: str):
        """Non-empty document produces at least one chunk."""
        assume(text.strip())

        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)
        doc = make_document(text)

        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1


class TestRecursiveChunkerSizeConstraint:
    """Test that RecursiveChunker produces non-empty, meaningful chunks."""

    @given(text=document_text(min_chars=100, max_chars=2000))
    @settings(max_examples=50)
    def test_chunks_are_non_empty(self, text: str):
        """All chunks have non-empty content.

        Note: RecursiveChunker does not guarantee strict size limits.
        It prioritizes semantic boundaries (paragraphs, sentences) over
        size constraints, and adds overlap which can increase chunk size.

        This test verifies the core property that all produced chunks
        have meaningful content.
        """
        assume(text.strip())
        chunk_size = 200
        chunk_overlap = 50

        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc = make_document(text)

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.content.strip(), "Chunk should have non-empty content"
            assert len(chunk.content) > 0

    @given(text=document_text(min_chars=500, max_chars=2000))
    @settings(max_examples=30)
    def test_multiple_chunks_for_large_text(self, text: str):
        """Large text produces multiple chunks, not one giant chunk."""
        assume(text.strip())

        # Use a small chunk size
        chunk_size = 100
        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=20)
        doc = make_document(text)

        chunks = chunker.chunk(doc)

        # For text 5x+ chunk_size, we should get multiple chunks
        if len(text.strip()) >= chunk_size * 5:
            assert len(chunks) >= 2, "Large text should produce multiple chunks"


class TestRecursiveChunkerSequentialIndices:
    """Test that chunk_index is sequential."""

    @given(text=document_text(min_chars=100, max_chars=2000))
    @settings(max_examples=50)
    def test_sequential_chunk_indices(self, text: str):
        """chunk_index values are 0, 1, 2, ... in order."""
        assume(text.strip())

        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        doc = make_document(text)

        chunks = chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestRecursiveChunkerSequentialIds:
    """Test that chunk.id follows doc_id:N pattern."""

    @given(text=document_text(min_chars=100, max_chars=2000))
    @settings(max_examples=50)
    def test_sequential_chunk_ids(self, text: str):
        """chunk.id follows 'doc_id:N' pattern."""
        assume(text.strip())

        doc_id = "recursive_test"
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        doc = make_document(text, doc_id=doc_id)

        chunks = chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            expected_id = f"{doc_id}:{i}"
            assert chunk.id == expected_id


class TestRecursiveChunkerDeterminism:
    """Test that same input produces same output."""

    @given(text=document_text(min_chars=50, max_chars=1000))
    @settings(max_examples=30)
    def test_deterministic_output(self, text: str):
        """Same input produces identical chunks."""
        assume(text.strip())

        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        doc = make_document(text)

        chunks1 = chunker.chunk(doc)
        chunks2 = chunker.chunk(doc)

        assert len(chunks1) == len(chunks2)

        for c1, c2 in zip(chunks1, chunks2):
            assert c1.id == c2.id
            assert c1.content == c2.content
            assert c1.chunk_index == c2.chunk_index


class TestRecursiveChunkerIdDeterminism:
    """Test that chunker_id is deterministic."""

    @given(
        chunk_size=st.integers(min_value=50, max_value=500),
        chunk_overlap=st.integers(min_value=0, max_value=49),
    )
    def test_chunker_id_deterministic(self, chunk_size: int, chunk_overlap: int):
        """Same params produce same chunker_id."""
        assume(chunk_overlap < chunk_size)

        chunker1 = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunker2 = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        assert chunker1.chunker_id == chunker2.chunker_id

    @given(
        chunk_size=st.integers(min_value=50, max_value=500),
        chunk_overlap=st.integers(min_value=0, max_value=49),
    )
    def test_chunker_id_format(self, chunk_size: int, chunk_overlap: int):
        """chunker_id follows 'recursive:size:overlap' format."""
        assume(chunk_overlap < chunk_size)

        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        expected = f"recursive:{chunk_size}:{chunk_overlap}"
        assert chunker.chunker_id == expected


# =============================================================================
# Empty Document Tests (Both Chunkers)
# =============================================================================


class TestEmptyDocumentHandling:
    """Test that empty documents produce no chunks."""

    def test_simple_chunker_empty_text(self):
        """SimpleChunker returns empty list for empty text."""
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=0)
        doc = make_document("")

        chunks = chunker.chunk(doc)

        assert chunks == []

    def test_simple_chunker_whitespace_only(self):
        """SimpleChunker returns empty list for whitespace-only text."""
        chunker = SimpleChunker(chunk_size=100, chunk_overlap=0)
        doc = make_document("   \n\n   \t   ")

        chunks = chunker.chunk(doc)

        assert chunks == []

    def test_recursive_chunker_empty_text(self):
        """RecursiveChunker returns empty list for empty text."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=50)
        doc = make_document("")

        chunks = chunker.chunk(doc)

        assert chunks == []

    def test_recursive_chunker_whitespace_only(self):
        """RecursiveChunker returns empty list for whitespace-only text."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=50)
        doc = make_document("   \n\n   \t   ")

        chunks = chunker.chunk(doc)

        assert chunks == []


# =============================================================================
# Chunk Metadata Tests
# =============================================================================


class TestChunkMetadata:
    """Test that chunks have correct metadata."""

    @given(text=document_text(min_chars=50, max_chars=500))
    @settings(max_examples=30)
    def test_simple_chunker_metadata(self, text: str):
        """SimpleChunker includes source_file and doc_id in metadata."""
        assume(text.strip())

        doc_id = "meta_test"
        chunker = SimpleChunker(chunk_size=200, chunk_overlap=0)
        doc = make_document(text, doc_id=doc_id)

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert "source_file" in chunk.metadata
            assert "doc_id" in chunk.metadata
            assert chunk.metadata["doc_id"] == doc_id
            assert chunk.doc_id == doc_id

    @given(text=document_text(min_chars=50, max_chars=500))
    @settings(max_examples=30)
    def test_recursive_chunker_metadata(self, text: str):
        """RecursiveChunker includes source_file and doc_id in metadata."""
        assume(text.strip())

        doc_id = "recursive_meta"
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        doc = make_document(text, doc_id=doc_id)

        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert "source_file" in chunk.metadata
            assert "doc_id" in chunk.metadata
            assert chunk.metadata["doc_id"] == doc_id
            assert chunk.doc_id == doc_id
