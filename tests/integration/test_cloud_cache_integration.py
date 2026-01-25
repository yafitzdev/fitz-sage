# tests/integration/test_cloud_cache_integration.py
"""
Integration tests for cloud cache in RAGPipeline.

Tests the full cache flow from end to end with mocked CloudClient.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from fitz_ai.cloud.cache_key import CacheVersions
from fitz_ai.core import Answer, Provenance
from fitz_ai.engines.fitz_rag.config.schema import PluginKwargs
from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import RGSAnswer, RGSSourceRef
from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline


@pytest.fixture
def mock_config():
    """Create mock FitzRagConfig using flat schema with real PluginKwargs."""
    config = Mock()

    # Core plugins (string format in new flat schema)
    config.chat = "openai"
    config.embedding = "openai"
    config.vector_db = "qdrant"

    # Plugin kwargs (real PluginKwargs, not dicts)
    config.chat_kwargs = PluginKwargs(model="gpt-4")
    config.embedding_kwargs = PluginKwargs()
    config.vector_db_kwargs = PluginKwargs()
    config.rerank_kwargs = PluginKwargs()
    config.vision_kwargs = PluginKwargs()

    # Optional plugins (None = disabled)
    config.rerank = None
    config.vision = None

    # Retrieval settings (flat)
    config.retrieval_plugin = "dense"
    config.collection = "test_collection"
    config.top_k = 5
    config.fetch_artifacts = False

    # RGS settings (flattened)
    config.enable_citations = True
    config.strict_grounding = True
    config.max_chunks = 8
    config.max_answer_chars = None
    config.include_query_in_context = True

    return config


@pytest.fixture
def mock_chunks():
    """Create mock chunks using real Chunk objects."""
    from fitz_ai.core import Chunk

    chunks = []
    for i in range(3):
        chunk = Chunk(
            id=f"chunk_{i}",
            doc_id=f"doc_{i}",
            content=f"Content {i}",
            chunk_index=i,
            metadata={"source": f"file_{i}.txt"},
        )
        chunks.append(chunk)
    return chunks


class TestFullCacheFlow:
    """Test the complete cache flow: miss → store → hit."""

    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_vector_db_plugin")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_chat_factory")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_llm_plugin")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_retrieval_plugin")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.create_matcher_from_store")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_table_store")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.EntityGraphStore")
    def test_cache_miss_then_store(
        self,
        mock_entity_graph,
        mock_table_store,
        mock_create_matcher,
        mock_get_retrieval,
        mock_get_llm,
        mock_get_chat_factory,
        mock_get_vector_db,
        mock_config,
        mock_chunks,
    ):
        """Test cache miss leads to full pipeline execution and storage."""
        # Setup mocks
        mock_vector_client = Mock()
        mock_get_vector_db.return_value = mock_vector_client

        mock_chat = Mock()
        mock_chat.plugin_name = "openai"
        mock_chat.params = {"model": "gpt-4"}
        mock_chat.chat.return_value = "This is a generated answer."

        mock_embedder = Mock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        # Chat factory returns the mock chat client for any tier
        def mock_factory(tier="fast"):
            return mock_chat

        mock_get_chat_factory.return_value = mock_factory
        mock_get_llm.return_value = mock_embedder

        mock_retrieval = Mock()
        mock_retrieval.collection = "test_collection"
        mock_retrieval.retrieve.return_value = mock_chunks
        mock_get_retrieval.return_value = mock_retrieval

        mock_create_matcher.return_value = None
        mock_table_store.return_value = Mock()
        mock_entity_graph.return_value = None

        # Create mock cloud client
        mock_cloud_client = Mock()
        mock_lookup_result = Mock()
        mock_lookup_result.hit = False  # Cache miss
        mock_cloud_client.lookup_cache.return_value = mock_lookup_result
        mock_cloud_client.store_cache.return_value = True

        # Create pipeline
        pipeline = RAGPipeline.from_config(mock_config, cloud_client=mock_cloud_client)

        # Mock RGS methods
        pipeline.rgs.build_prompt = Mock(return_value=Mock(system="System", user="User"))
        pipeline.rgs.build_answer = Mock(
            return_value=RGSAnswer(
                answer="Generated answer",
                sources=[
                    RGSSourceRef(
                        source_id="source_1",
                        index=0,
                        doc_id="doc_1",
                        content="Content 1",
                        metadata={},
                    )
                ],
                mode=None,
            )
        )

        # Run query
        result = pipeline.run("What is quantum computing?")

        # Verify cache lookup was called
        assert mock_cloud_client.lookup_cache.called

        # Verify full pipeline executed (cache miss)
        assert mock_retrieval.retrieve.called
        assert mock_chat.chat.called

        # Verify cache storage was called
        assert mock_cloud_client.store_cache.called

        # Verify result
        assert result.answer == "Generated answer"

    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_vector_db_plugin")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_chat_factory")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_llm_plugin")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_retrieval_plugin")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.create_matcher_from_store")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_table_store")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.EntityGraphStore")
    def test_cache_hit_skips_pipeline(
        self,
        mock_entity_graph,
        mock_table_store,
        mock_create_matcher,
        mock_get_retrieval,
        mock_get_llm,
        mock_get_chat_factory,
        mock_get_vector_db,
        mock_config,
        mock_chunks,
    ):
        """Test cache hit skips full pipeline execution."""
        # Setup mocks
        mock_vector_client = Mock()
        mock_get_vector_db.return_value = mock_vector_client

        mock_chat = Mock()
        mock_embedder = Mock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        # Chat factory returns the mock chat client for any tier
        def mock_factory(tier="fast"):
            return mock_chat

        mock_get_chat_factory.return_value = mock_factory
        mock_get_llm.return_value = mock_embedder

        mock_retrieval = Mock()
        mock_retrieval.collection = "test_collection"
        mock_retrieval.retrieve.return_value = mock_chunks
        mock_get_retrieval.return_value = mock_retrieval

        mock_create_matcher.return_value = None
        mock_table_store.return_value = Mock()
        mock_entity_graph.return_value = None

        # Create mock cloud client with cached answer
        mock_cloud_client = Mock()
        cached_answer = Answer(
            text="Cached answer from cloud",
            provenance=[
                Provenance(
                    source_id="cached_source",
                    excerpt="Cached content",
                    metadata={},
                )
            ],
            mode=None,
            metadata={},
        )

        mock_lookup_result = Mock()
        mock_lookup_result.hit = True  # Cache hit!
        mock_lookup_result.answer = cached_answer
        mock_cloud_client.lookup_cache.return_value = mock_lookup_result

        # Create pipeline
        pipeline = RAGPipeline.from_config(mock_config, cloud_client=mock_cloud_client)

        # Run query
        result = pipeline.run("What is quantum computing?")

        # Verify cache lookup was called
        assert mock_cloud_client.lookup_cache.called

        # Verify pipeline was NOT executed (cache hit)
        assert mock_retrieval.retrieve.called  # Still retrieves for cache key
        assert not mock_chat.chat.called  # Should NOT call LLM

        # Verify cache storage was NOT called (cache hit)
        assert not mock_cloud_client.store_cache.called

        # Verify result from cache
        assert result.answer == "Cached answer from cloud"

    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_vector_db_plugin")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_chat_factory")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_llm_plugin")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_retrieval_plugin")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.create_matcher_from_store")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.get_table_store")
    @patch("fitz_ai.engines.fitz_rag.pipeline.engine.EntityGraphStore")
    def test_embedding_reuse_optimization(
        self,
        mock_entity_graph,
        mock_table_store,
        mock_create_matcher,
        mock_get_retrieval,
        mock_get_llm,
        mock_get_chat_factory,
        mock_get_vector_db,
        mock_config,
        mock_chunks,
    ):
        """Test that cached query embedding is reused in storage."""
        # Setup mocks
        mock_vector_client = Mock()
        mock_get_vector_db.return_value = mock_vector_client

        mock_chat = Mock()
        mock_chat.plugin_name = "openai"
        mock_chat.params = {"model": "gpt-4"}
        mock_chat.chat.return_value = "Generated answer"

        mock_embedder = Mock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        # Chat factory returns the mock chat client for any tier
        def mock_factory(tier="fast"):
            return mock_chat

        mock_get_chat_factory.return_value = mock_factory
        mock_get_llm.return_value = mock_embedder

        mock_retrieval = Mock()
        mock_retrieval.collection = "test_collection"
        mock_retrieval.retrieve.return_value = mock_chunks
        mock_get_retrieval.return_value = mock_retrieval

        mock_create_matcher.return_value = None
        mock_table_store.return_value = Mock()
        mock_entity_graph.return_value = None

        # Create mock cloud client (cache miss)
        mock_cloud_client = Mock()
        mock_lookup_result = Mock()
        mock_lookup_result.hit = False
        mock_cloud_client.lookup_cache.return_value = mock_lookup_result
        mock_cloud_client.store_cache.return_value = True

        # Create pipeline
        pipeline = RAGPipeline.from_config(mock_config, cloud_client=mock_cloud_client)

        # Mock RGS methods
        pipeline.rgs.build_prompt = Mock(return_value=Mock(system="System", user="User"))
        pipeline.rgs.build_answer = Mock(
            return_value=RGSAnswer(
                answer="Generated answer",
                sources=[
                    RGSSourceRef(source_id="s", index=0, doc_id="d", content="c", metadata={})
                ],
                mode=None,
            )
        )

        # Run query
        pipeline.run("What is quantum computing?")

        # Verify cloud_client.store_cache was called
        assert mock_cloud_client.store_cache.called

        # Verify the cached embedding [0.1, 0.2, 0.3] was passed to store_cache
        store_call_kwargs = mock_cloud_client.store_cache.call_args.kwargs
        assert "query_embedding" in store_call_kwargs
        assert store_call_kwargs["query_embedding"] == [0.1, 0.2, 0.3]


class TestCacheKeyDeterminism:
    """Test that cache keys are deterministic."""

    def test_same_query_same_chunks_same_key(self):
        """Same query with same chunks should produce same cache key."""
        from fitz_ai.cloud.cache_key import compute_cache_key

        versions = CacheVersions(
            optimizer="1.0",
            engine="0.5.2",
            collection="abc123",
            llm_model="openai:gpt-4",
            prompt_template="default",
        )

        key1 = compute_cache_key("test query", "fingerprint123", versions)
        key2 = compute_cache_key("test query", "fingerprint123", versions)

        assert key1 == key2

    def test_different_query_different_key(self):
        """Different queries should produce different cache keys."""
        from fitz_ai.cloud.cache_key import compute_cache_key

        versions = CacheVersions(
            optimizer="1.0",
            engine="0.5.2",
            collection="abc123",
            llm_model="openai:gpt-4",
            prompt_template="default",
        )

        key1 = compute_cache_key("query 1", "fingerprint123", versions)
        key2 = compute_cache_key("query 2", "fingerprint123", versions)

        assert key1 != key2

    def test_different_chunks_different_key(self):
        """Different chunks should produce different cache keys."""
        from fitz_ai.cloud.cache_key import compute_cache_key

        versions = CacheVersions(
            optimizer="1.0",
            engine="0.5.2",
            collection="abc123",
            llm_model="openai:gpt-4",
            prompt_template="default",
        )

        key1 = compute_cache_key("test query", "fingerprint_a", versions)
        key2 = compute_cache_key("test query", "fingerprint_b", versions)

        assert key1 != key2

    def test_different_version_different_key(self):
        """Different versions should produce different cache keys."""
        from fitz_ai.cloud.cache_key import compute_cache_key

        versions1 = CacheVersions(
            optimizer="1.0",
            engine="0.5.2",
            collection="abc123",
            llm_model="openai:gpt-4",
            prompt_template="default",
        )

        versions2 = CacheVersions(
            optimizer="1.0",
            engine="0.5.2",
            collection="xyz789",  # Different collection
            llm_model="openai:gpt-4",
            prompt_template="default",
        )

        key1 = compute_cache_key("test query", "fingerprint123", versions1)
        key2 = compute_cache_key("test query", "fingerprint123", versions2)

        assert key1 != key2
