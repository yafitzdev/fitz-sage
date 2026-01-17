# tests/unit/integrations/test_base_optimizer.py
"""Tests for base optimizer."""

from unittest.mock import MagicMock, patch

import pytest

from fitz_ai.integrations.base import FitzOptimizer, OptimizationResult


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_hit_result(self):
        """Hit result has answer and sources."""
        result = OptimizationResult(
            hit=True,
            answer="Test answer",
            sources=[{"source_id": "doc1", "excerpt": "text"}],
        )
        assert result.hit is True
        assert result.answer == "Test answer"
        assert len(result.sources) == 1

    def test_miss_result(self):
        """Miss result has routing advice."""
        result = OptimizationResult(
            hit=False,
            routing_advice={"complexity": "simple", "recommended_model": "fast"},
        )
        assert result.hit is False
        assert result.answer is None
        assert result.routing_advice["complexity"] == "simple"


class TestFitzOptimizer:
    """Tests for FitzOptimizer class."""

    def test_generate_org_id(self):
        """Generates deterministic org_id from API key."""
        org_id1 = FitzOptimizer._generate_org_id("fitz_abc123456789012345")
        org_id2 = FitzOptimizer._generate_org_id("fitz_abc123456789012345")
        org_id3 = FitzOptimizer._generate_org_id("fitz_different_key12345")

        # Same key = same org_id
        assert org_id1 == org_id2

        # Different key = different org_id
        assert org_id1 != org_id3

        # Valid UUID format
        assert len(org_id1) == 36  # UUID format

    @patch("fitz_ai.integrations.base.CloudClient")
    def test_lookup_dimension_validation(self, mock_client_class):
        """Wrong embedding dimension returns miss."""
        optimizer = FitzOptimizer(
            api_key="fitz_test_key_12345",
            org_key="a" * 64,
        )

        # Wrong dimension (768 instead of 1536)
        result = optimizer.lookup(
            query="test query",
            query_embedding=[1.0] * 768,
            chunk_ids=["c1", "c2"],
            llm_model="gpt-4o",
        )

        assert result.hit is False
        assert result.answer is None
        # Client lookup should not be called with wrong dimension
        mock_client_class.return_value.lookup_cache.assert_not_called()

    @patch("fitz_ai.integrations.base.CloudClient")
    def test_lookup_cache_hit(self, mock_client_class):
        """Cache hit returns decrypted answer."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock cache hit
        mock_result = MagicMock()
        mock_result.hit = True
        mock_result.answer = MagicMock()
        mock_result.answer.text = "Cached answer"
        mock_result.answer.provenance = []
        mock_client.lookup_cache.return_value = mock_result

        optimizer = FitzOptimizer(
            api_key="fitz_test_key_12345",
            org_key="a" * 64,
        )

        result = optimizer.lookup(
            query="test query",
            query_embedding=[1.0] * 1536,
            chunk_ids=["c1", "c2"],
            llm_model="gpt-4o",
        )

        assert result.hit is True
        assert result.answer == "Cached answer"
        mock_client.lookup_cache.assert_called_once()

    @patch("fitz_ai.integrations.base.CloudClient")
    def test_lookup_cache_miss_with_routing(self, mock_client_class):
        """Cache miss returns routing advice."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock cache miss with routing
        mock_result = MagicMock()
        mock_result.hit = False
        mock_result.answer = None
        mock_result.routing = MagicMock()
        mock_result.routing.recommended_model = "fast"
        mock_result.routing.complexity = "simple"
        mock_result.routing.dedup_chunks = [1, 3]
        mock_client.lookup_cache.return_value = mock_result

        optimizer = FitzOptimizer(
            api_key="fitz_test_key_12345",
            org_key="a" * 64,
        )

        result = optimizer.lookup(
            query="test query",
            query_embedding=[1.0] * 1536,
            chunk_ids=["c1", "c2"],
            llm_model="gpt-4o",
        )

        assert result.hit is False
        assert result.routing_advice is not None
        assert result.routing_advice["recommended_model"] == "fast"
        assert result.routing_advice["complexity"] == "simple"
        assert result.routing_advice["dedup_chunks"] == [1, 3]

    @patch("fitz_ai.integrations.base.CloudClient")
    def test_store_dimension_validation(self, mock_client_class):
        """Wrong embedding dimension fails silently."""
        optimizer = FitzOptimizer(
            api_key="fitz_test_key_12345",
            org_key="a" * 64,
        )

        # Wrong dimension
        stored = optimizer.store(
            query="test query",
            query_embedding=[1.0] * 768,
            chunk_ids=["c1", "c2"],
            llm_model="gpt-4o",
            answer_text="Test answer",
        )

        assert stored is False
        mock_client_class.return_value.store_cache.assert_not_called()

    @patch("fitz_ai.integrations.base.CloudClient")
    def test_store_success(self, mock_client_class):
        """Successful store returns True."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.store_cache.return_value = True

        optimizer = FitzOptimizer(
            api_key="fitz_test_key_12345",
            org_key="a" * 64,
        )

        stored = optimizer.store(
            query="test query",
            query_embedding=[1.0] * 1536,
            chunk_ids=["c1", "c2"],
            llm_model="gpt-4o",
            answer_text="Test answer",
            sources=[{"source_id": "doc1", "excerpt": "text"}],
        )

        assert stored is True
        mock_client.store_cache.assert_called_once()

    @patch("fitz_ai.integrations.base.CloudClient")
    def test_embed_query_no_function(self, mock_client_class):
        """No embedding function returns None."""
        optimizer = FitzOptimizer(
            api_key="fitz_test_key_12345",
            org_key="a" * 64,
            embedding_fn=None,
        )

        result = optimizer.embed_query("test query")
        assert result is None

    @patch("fitz_ai.integrations.base.CloudClient")
    def test_embed_query_success(self, mock_client_class):
        """Successful embedding returns vector."""
        mock_embed_fn = MagicMock(return_value=[1.0] * 1536)

        optimizer = FitzOptimizer(
            api_key="fitz_test_key_12345",
            org_key="a" * 64,
            embedding_fn=mock_embed_fn,
        )

        result = optimizer.embed_query("test query")

        assert result is not None
        assert len(result) == 1536
        mock_embed_fn.assert_called_once_with("test query")

    @patch("fitz_ai.integrations.base.CloudClient")
    def test_context_manager(self, mock_client_class):
        """Context manager closes client."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with FitzOptimizer(
            api_key="fitz_test_key_12345",
            org_key="a" * 64,
        ) as optimizer:
            pass

        mock_client.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
