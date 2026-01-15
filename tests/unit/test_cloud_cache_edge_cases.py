# tests/unit/test_cloud_cache_edge_cases.py
"""
Edge case tests for cloud cache functionality.

Tests error handling, fail-open behavior, and corner cases.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import RGSAnswer, RGSSourceRef


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


class TestCacheDisabled:
    """Test behavior when cache is disabled or unavailable."""

    def test_no_cloud_client_skips_cache(self, mock_pipeline):
        """Pipeline should work normally without cloud_client."""
        mock_pipeline.cloud_client = None

        chunks = [Mock(id="chunk_1")]
        result = mock_pipeline._check_cloud_cache("test query", chunks)

        assert result is None

    def test_no_embedder_skips_cache(self, mock_pipeline):
        """Pipeline should work normally without embedder."""
        mock_pipeline.embedder = None

        chunks = [Mock(id="chunk_1")]
        result = mock_pipeline._check_cloud_cache("test query", chunks)

        assert result is None


class TestEmptyChunks:
    """Test behavior with empty or no chunks."""

    def test_empty_chunks_list(self, mock_pipeline):
        """Should handle empty chunks list gracefully."""
        mock_lookup_result = Mock()
        mock_lookup_result.hit = False
        mock_pipeline.cloud_client.lookup_cache.return_value = mock_lookup_result

        result = mock_pipeline._check_cloud_cache("test query", [])

        # Should compute fingerprint from empty list (deterministic)
        assert result is None

    def test_storage_with_empty_chunks(self, mock_pipeline):
        """Should handle storage with empty chunks."""
        mock_pipeline.cloud_client.store_cache.return_value = True

        rgs_answer = RGSAnswer(
            answer="Answer",
            sources=[],
            mode=None,
        )

        # Should not raise exception
        mock_pipeline._store_in_cloud_cache("test query", [], rgs_answer)


class TestCacheApiErrors:
    """Test fail-open behavior when cache API has errors."""

    def test_lookup_error_returns_none(self, mock_pipeline):
        """Cache lookup errors should return None, not raise."""
        mock_pipeline.cloud_client.lookup_cache.side_effect = Exception("API error")

        chunks = [Mock(id="chunk_1")]
        result = mock_pipeline._check_cloud_cache("test query", chunks)

        # Should return None and continue with pipeline
        assert result is None

    def test_lookup_network_timeout(self, mock_pipeline):
        """Network timeouts should be handled gracefully."""
        import httpx

        mock_pipeline.cloud_client.lookup_cache.side_effect = httpx.TimeoutException(
            "Request timed out"
        )

        chunks = [Mock(id="chunk_1")]
        result = mock_pipeline._check_cloud_cache("test query", chunks)

        assert result is None

    def test_storage_error_does_not_raise(self, mock_pipeline):
        """Cache storage errors should not break the pipeline."""
        mock_pipeline.cloud_client.store_cache.side_effect = Exception("Storage failed")

        rgs_answer = RGSAnswer(
            answer="Answer",
            sources=[RGSSourceRef(source_id="s", index=0, doc_id="d", content="c", metadata={})],
            mode=None,
        )

        chunks = [Mock(id="chunk_1")]

        # Should not raise exception
        mock_pipeline._store_in_cloud_cache("test query", chunks, rgs_answer)


class TestCollectionVersionComputation:
    """Test edge cases in collection version computation."""

    def test_no_active_files(self, mock_pipeline):
        """Should handle collection with no active files."""
        with patch("fitz_ai.ingestion.state.IngestStateManager") as mock_manager:
            mock_state = Mock()
            mock_root = Mock()
            mock_root.files = {}  # No files
            mock_state.roots = {"root1": mock_root}

            mock_manager_instance = Mock()
            mock_manager_instance.state = mock_state
            mock_manager.return_value = mock_manager_instance

            version = mock_pipeline._get_collection_version()

            # Should return valid hash even with no files
            assert version is not None
            assert len(version) == 16

    def test_collection_not_found(self, mock_pipeline):
        """Should handle case where collection doesn't exist."""
        with patch("fitz_ai.ingestion.state.IngestStateManager") as mock_manager:
            mock_state = Mock()
            mock_state.roots = {}  # No roots

            mock_manager_instance = Mock()
            mock_manager_instance.state = mock_state
            mock_manager.return_value = mock_manager_instance

            version = mock_pipeline._get_collection_version()

            # Should return valid hash (empty state)
            assert version is not None

    def test_ingest_manager_load_fails(self, mock_pipeline):
        """Should return 'unknown' if IngestStateManager fails."""
        with patch("fitz_ai.ingestion.state.IngestStateManager") as mock_manager:
            mock_manager_instance = Mock()
            mock_manager_instance.load.side_effect = FileNotFoundError("No ingest.json")
            mock_manager.return_value = mock_manager_instance

            version = mock_pipeline._get_collection_version()

            assert version == "unknown"


class TestLlmModelIdExtraction:
    """Test edge cases in LLM model ID extraction."""

    def test_chat_client_no_plugin_name(self, mock_pipeline):
        """Should handle chat client without plugin_name."""
        delattr(mock_pipeline.chat, "plugin_name")
        mock_pipeline.chat.params = {"model": "gpt-4"}

        model_id = mock_pipeline._get_llm_model_id()

        assert model_id == "unknown"

    def test_chat_client_missing_model_key(self, mock_pipeline):
        """Should handle params dict without 'model' key."""
        mock_pipeline.chat.params = {"temperature": 0.7}  # No model

        model_id = mock_pipeline._get_llm_model_id()

        assert model_id == "unknown"

    def test_chat_client_none_model(self, mock_pipeline):
        """Should handle None model value."""
        mock_pipeline.chat.params = {"model": None}

        model_id = mock_pipeline._get_llm_model_id()

        # Should return "unknown" since None is not a valid model
        assert model_id == "unknown"


class TestAnswerConversion:
    """Test edge cases in Answer ↔ RGSAnswer conversion."""

    def test_convert_answer_with_empty_excerpt(self, mock_pipeline):
        """Should handle provenance with empty excerpts."""
        from fitz_ai.core import Answer, Provenance

        answer = Answer(
            text="Test answer",
            provenance=[
                Provenance(
                    source_id="source_1",
                    excerpt="",  # Empty excerpt
                    metadata={},
                )
            ],
            mode=None,
            metadata={},
        )

        rgs_answer = mock_pipeline._answer_to_rgs_answer(answer)

        assert len(rgs_answer.sources) == 1
        assert rgs_answer.sources[0].content == ""

    def test_convert_answer_with_none_excerpt(self, mock_pipeline):
        """Should handle provenance with None excerpts."""
        from fitz_ai.core import Answer, Provenance

        answer = Answer(
            text="Test answer",
            provenance=[
                Provenance(
                    source_id="source_1",
                    excerpt=None,  # None excerpt
                    metadata={},
                )
            ],
            mode=None,
            metadata={},
        )

        rgs_answer = mock_pipeline._answer_to_rgs_answer(answer)

        assert len(rgs_answer.sources) == 1
        assert rgs_answer.sources[0].content == ""


class TestEmbeddingCache:
    """Test query embedding caching behavior."""

    def test_embedding_computed_for_each_query(self, mock_pipeline):
        """Embedding should be computed for each query (overwritten, not reused across queries)."""
        mock_lookup_result = Mock()
        mock_lookup_result.hit = False
        mock_pipeline.cloud_client.lookup_cache.return_value = mock_lookup_result

        chunks = [Mock(id="chunk_1")]

        # First query
        mock_pipeline._check_cloud_cache("query 1", chunks)
        first_embedding = mock_pipeline._cached_query_embedding

        # Second query
        mock_pipeline._check_cloud_cache("query 2", chunks)
        second_embedding = mock_pipeline._cached_query_embedding

        # Both should have cached embeddings
        assert first_embedding is not None
        assert second_embedding is not None

        # Should be different calls (not reused across queries)
        assert mock_pipeline.embedder.embed.call_count == 2

    def test_storage_without_prior_lookup(self, mock_pipeline):
        """Storage should work even if lookup wasn't called (computes embedding)."""
        # No cached embedding
        assert not hasattr(mock_pipeline, "_cached_query_embedding")

        mock_pipeline.cloud_client.store_cache.return_value = True

        rgs_answer = RGSAnswer(
            answer="Answer",
            sources=[RGSSourceRef(source_id="s", index=0, doc_id="d", content="c", metadata={})],
            mode=None,
        )

        chunks = [Mock(id="chunk_1")]

        # Should not raise exception
        mock_pipeline._store_in_cloud_cache("query", chunks, rgs_answer)

        # Should have computed embedding
        mock_pipeline.embedder.embed.assert_called_once()


class TestRetrievalFingerprintEdgeCases:
    """Test edge cases in retrieval fingerprint computation."""

    def test_fingerprint_deterministic_with_shuffled_ids(self):
        """Fingerprint should be deterministic regardless of chunk order."""
        from fitz_ai.cloud.cache_key import compute_retrieval_fingerprint

        # Same chunk IDs in different order
        fp1 = compute_retrieval_fingerprint(["chunk_1", "chunk_2", "chunk_3"])
        fp2 = compute_retrieval_fingerprint(["chunk_3", "chunk_1", "chunk_2"])

        # Should be identical (sorted internally)
        assert fp1 == fp2

    def test_fingerprint_with_single_chunk(self):
        """Should handle single chunk."""
        from fitz_ai.cloud.cache_key import compute_retrieval_fingerprint

        fp = compute_retrieval_fingerprint(["chunk_1"])

        assert fp is not None
        assert isinstance(fp, str)

    def test_fingerprint_with_empty_list(self):
        """Should handle empty chunk list."""
        from fitz_ai.cloud.cache_key import compute_retrieval_fingerprint

        fp = compute_retrieval_fingerprint([])

        # Should return valid hash (of empty string)
        assert fp is not None
        assert isinstance(fp, str)
