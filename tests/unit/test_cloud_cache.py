# tests/unit/test_cloud_cache.py
"""
Unit tests for cloud cache functionality in RAGPipeline.

Tests the cache helper methods and cache operations in isolation.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from fitz_ai.core import Answer, Provenance
from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import RGSAnswer, RGSSourceRef
from fitz_ai.engines.fitz_rag.pipeline.engine import CLOUD_OPTIMIZER_VERSION


@pytest.fixture
def mock_pipeline():
    """Create a mock RAGPipeline with required attributes."""
    from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline

    pipeline = RAGPipeline.__new__(RAGPipeline)

    # Mock retrieval with collection attribute
    pipeline.retrieval = Mock()
    pipeline.retrieval.collection = "test_collection"

    # Mock chat client with params
    pipeline.chat = Mock()
    pipeline.chat.plugin_name = "openai"
    pipeline.chat.params = {"model": "gpt-4"}

    # Mock embedder
    pipeline.embedder = Mock()
    pipeline.embedder.embed = Mock(return_value=[0.1, 0.2, 0.3])

    # Mock cloud client
    pipeline.cloud_client = Mock()

    return pipeline


@pytest.fixture
def mock_chunks():
    """Create mock chunks for testing."""
    chunks = []
    for i in range(3):
        chunk = Mock()
        chunk.id = f"chunk_{i}"
        chunk.doc_id = f"doc_{i}"
        chunk.content = f"Content {i}"
        chunk.metadata = {"source": f"file_{i}.txt"}
        chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_rgs_answer():
    """Create mock RGSAnswer for testing."""
    sources = [
        RGSSourceRef(
            source_id="source_1",
            index=0,
            doc_id="doc_1",
            content="Test content 1",
            metadata={"file": "test1.txt"},
        ),
        RGSSourceRef(
            source_id="source_2",
            index=1,
            doc_id="doc_2",
            content="Test content 2",
            metadata={"file": "test2.txt"},
        ),
    ]
    return RGSAnswer(
        answer="This is a test answer.",
        sources=sources,
        mode=None,
    )


class TestGetCollectionVersion:
    """Test _get_collection_version() method."""

    def test_collection_version_deterministic(self, mock_pipeline):
        """Collection version should be deterministic for same state."""
        with patch("fitz_ai.ingestion.state.IngestStateManager") as mock_manager:
            # Setup mock state
            mock_state = Mock()
            mock_root = Mock()
            mock_file1 = Mock()
            mock_file1.is_active.return_value = True
            mock_file1.collection = "test_collection"
            mock_file1.content_hash = "hash1"
            mock_file1.chunker_id = "chunker1"
            mock_file1.parser_id = "parser1"
            mock_file1.embedding_id = "embed1"

            mock_root.files = {"file1": mock_file1}
            mock_state.roots = {"root1": mock_root}

            mock_manager_instance = Mock()
            mock_manager_instance.state = mock_state
            mock_manager.return_value = mock_manager_instance

            # First call
            version1 = mock_pipeline._get_collection_version()

            # Second call (should be same)
            version2 = mock_pipeline._get_collection_version()

            assert version1 == version2
            assert len(version1) == 16  # Should be truncated to 16 chars

    def test_collection_version_filters_by_collection(self, mock_pipeline):
        """Should only hash files from the specified collection."""
        with patch("fitz_ai.ingestion.state.IngestStateManager") as mock_manager:
            # Setup state with multiple collections
            mock_state = Mock()
            mock_root = Mock()

            mock_file1 = Mock()
            mock_file1.is_active.return_value = True
            mock_file1.collection = "test_collection"
            mock_file1.content_hash = "hash1"
            mock_file1.chunker_id = "chunker1"
            mock_file1.parser_id = "parser1"
            mock_file1.embedding_id = "embed1"

            mock_file2 = Mock()
            mock_file2.is_active.return_value = True
            mock_file2.collection = "other_collection"  # Different collection
            mock_file2.content_hash = "hash2"
            mock_file2.chunker_id = "chunker2"
            mock_file2.parser_id = "parser2"
            mock_file2.embedding_id = "embed2"

            mock_root.files = {"file1": mock_file1, "file2": mock_file2}
            mock_state.roots = {"root1": mock_root}

            mock_manager_instance = Mock()
            mock_manager_instance.state = mock_state
            mock_manager.return_value = mock_manager_instance

            version = mock_pipeline._get_collection_version()

            # Version should only include file1 (test_collection)
            assert version is not None

    def test_collection_version_handles_exception(self, mock_pipeline):
        """Should return 'unknown' if version computation fails."""
        with patch(
            "fitz_ai.ingestion.state.IngestStateManager",
            side_effect=Exception("Test error"),
        ):
            version = mock_pipeline._get_collection_version()
            assert version == "unknown"


class TestGetLlmModelId:
    """Test _get_llm_model_id() method."""

    def test_extracts_model_from_params(self, mock_pipeline):
        """Should extract plugin_name and model from chat client."""
        model_id = mock_pipeline._get_llm_model_id()
        assert model_id == "openai:gpt-4"

    def test_returns_unknown_if_no_params(self, mock_pipeline):
        """Should return 'unknown' if chat client has no params."""
        delattr(mock_pipeline.chat, "params")
        model_id = mock_pipeline._get_llm_model_id()
        assert model_id == "unknown"

    def test_returns_unknown_if_no_model_in_params(self, mock_pipeline):
        """Should return 'unknown' if params doesn't contain model."""
        mock_pipeline.chat.params = {}
        model_id = mock_pipeline._get_llm_model_id()
        assert model_id == "unknown"


class TestAnswerToRgsAnswer:
    """Test _answer_to_rgs_answer() method."""

    def test_converts_answer_to_rgs_answer(self, mock_pipeline):
        """Should convert Answer to RGSAnswer format."""
        answer = Answer(
            text="Test answer",
            provenance=[
                Provenance(
                    source_id="source_1",
                    excerpt="Excerpt 1",
                    metadata={"file": "test.txt"},
                )
            ],
            mode=None,
            metadata={},
        )

        rgs_answer = mock_pipeline._answer_to_rgs_answer(answer)

        assert isinstance(rgs_answer, RGSAnswer)
        assert rgs_answer.answer == "Test answer"
        assert len(rgs_answer.sources) == 1
        assert rgs_answer.sources[0].source_id == "source_1"
        assert rgs_answer.sources[0].content == "Excerpt 1"

    def test_handles_empty_provenance(self, mock_pipeline):
        """Should handle Answer with no provenance."""
        answer = Answer(
            text="Test answer",
            provenance=[],
            mode=None,
            metadata={},
        )

        rgs_answer = mock_pipeline._answer_to_rgs_answer(answer)

        assert rgs_answer.answer == "Test answer"
        assert len(rgs_answer.sources) == 0


class TestCheckCloudCache:
    """Test _check_cloud_cache() method."""

    def test_returns_none_if_no_cloud_client(self, mock_pipeline, mock_chunks):
        """Should return None if cloud_client is not available."""
        mock_pipeline.cloud_client = None
        result = mock_pipeline._check_cloud_cache("test query", mock_chunks)
        assert result is None

    def test_returns_none_if_no_embedder(self, mock_pipeline, mock_chunks):
        """Should return None if embedder is not available."""
        mock_pipeline.embedder = None
        result = mock_pipeline._check_cloud_cache("test query", mock_chunks)
        assert result is None

    def test_cache_hit_returns_rgs_answer(self, mock_pipeline, mock_chunks):
        """Should return RGSAnswer on cache hit."""
        # Mock cache hit
        cached_answer = Answer(
            text="Cached answer",
            provenance=[
                Provenance(
                    source_id="cached_source",
                    excerpt="Cached excerpt",
                    metadata={},
                )
            ],
            mode=None,
            metadata={},
        )

        mock_lookup_result = Mock()
        mock_lookup_result.hit = True
        mock_lookup_result.answer = cached_answer

        mock_pipeline.cloud_client.lookup_cache.return_value = mock_lookup_result

        result = mock_pipeline._check_cloud_cache("test query", mock_chunks)

        assert result is not None
        assert isinstance(result, RGSAnswer)
        assert result.answer == "Cached answer"

    def test_cache_miss_returns_none(self, mock_pipeline, mock_chunks):
        """Should return None on cache miss."""
        mock_lookup_result = Mock()
        mock_lookup_result.hit = False

        mock_pipeline.cloud_client.lookup_cache.return_value = mock_lookup_result

        result = mock_pipeline._check_cloud_cache("test query", mock_chunks)

        assert result is None

    def test_caches_query_embedding(self, mock_pipeline, mock_chunks):
        """Should cache query embedding for reuse in storage."""
        mock_lookup_result = Mock()
        mock_lookup_result.hit = False
        mock_pipeline.cloud_client.lookup_cache.return_value = mock_lookup_result

        mock_pipeline._check_cloud_cache("test query", mock_chunks)

        # Check that embedding was cached
        assert hasattr(mock_pipeline, "_cached_query_embedding")
        assert mock_pipeline._cached_query_embedding == [0.1, 0.2, 0.3]

    def test_uses_correct_cache_versions(self, mock_pipeline, mock_chunks):
        """Should use correct version info in cache key."""
        import fitz_ai

        mock_lookup_result = Mock()
        mock_lookup_result.hit = False
        mock_pipeline.cloud_client.lookup_cache.return_value = mock_lookup_result

        # Mock collection version
        mock_pipeline._collection_version = "abc123"

        mock_pipeline._check_cloud_cache("test query", mock_chunks)

        # Verify lookup_cache was called with correct versions
        call_args = mock_pipeline.cloud_client.lookup_cache.call_args
        versions = call_args.kwargs["versions"]

        assert versions.optimizer == CLOUD_OPTIMIZER_VERSION
        assert versions.engine == fitz_ai.__version__
        assert versions.collection == "abc123"
        assert versions.llm_model == "openai:gpt-4"
        assert versions.prompt_template == "default"

    def test_handles_exception_gracefully(self, mock_pipeline, mock_chunks):
        """Should return None and log warning on exception."""
        mock_pipeline.cloud_client.lookup_cache.side_effect = Exception("Test error")

        result = mock_pipeline._check_cloud_cache("test query", mock_chunks)

        assert result is None


class TestStoreInCloudCache:
    """Test _store_in_cloud_cache() method."""

    def test_returns_early_if_no_cloud_client(self, mock_pipeline, mock_chunks, mock_rgs_answer):
        """Should return early if cloud_client is not available."""
        mock_pipeline.cloud_client = None
        mock_pipeline._store_in_cloud_cache("test query", mock_chunks, mock_rgs_answer)
        # Should not raise exception

    def test_returns_early_if_no_embedder(self, mock_pipeline, mock_chunks, mock_rgs_answer):
        """Should return early if embedder is not available."""
        mock_pipeline.embedder = None
        mock_pipeline._store_in_cloud_cache("test query", mock_chunks, mock_rgs_answer)
        # Should not raise exception

    def test_reuses_cached_embedding(self, mock_pipeline, mock_chunks, mock_rgs_answer):
        """Should reuse cached query embedding instead of computing again."""
        # Set cached embedding
        mock_pipeline._cached_query_embedding = [0.5, 0.6, 0.7]

        mock_pipeline.cloud_client.store_cache.return_value = True

        mock_pipeline._store_in_cloud_cache("test query", mock_chunks, mock_rgs_answer)

        # Embedder.embed should NOT have been called
        mock_pipeline.embedder.embed.assert_not_called()

        # Verify store_cache was called with cached embedding
        call_args = mock_pipeline.cloud_client.store_cache.call_args
        assert call_args.kwargs["query_embedding"] == [0.5, 0.6, 0.7]

    def test_computes_embedding_if_not_cached(self, mock_pipeline, mock_chunks, mock_rgs_answer):
        """Should compute embedding if not cached."""
        # No cached embedding
        mock_pipeline.cloud_client.store_cache.return_value = True

        mock_pipeline._store_in_cloud_cache("test query", mock_chunks, mock_rgs_answer)

        # Embedder.embed should have been called
        mock_pipeline.embedder.embed.assert_called_once_with("test query")

    def test_converts_rgs_answer_to_answer(self, mock_pipeline, mock_chunks, mock_rgs_answer):
        """Should convert RGSAnswer to Answer for storage."""
        mock_pipeline.cloud_client.store_cache.return_value = True

        mock_pipeline._store_in_cloud_cache("test query", mock_chunks, mock_rgs_answer)

        # Verify store_cache was called with Answer object
        call_args = mock_pipeline.cloud_client.store_cache.call_args
        answer = call_args.kwargs["answer"]

        assert isinstance(answer, Answer)
        assert answer.text == "This is a test answer."
        assert len(answer.provenance) == 2
        assert answer.metadata["engine"] == "fitz_rag"

    def test_handles_exception_gracefully(self, mock_pipeline, mock_chunks, mock_rgs_answer):
        """Should log warning but not raise on exception."""
        mock_pipeline.cloud_client.store_cache.side_effect = Exception("Test error")

        # Should not raise exception
        mock_pipeline._store_in_cloud_cache("test query", mock_chunks, mock_rgs_answer)
