# tests/unit/integrations/test_llamaindex.py
"""Tests for LlamaIndex integration."""

from unittest.mock import MagicMock, patch

import pytest

# Skip if llama-index-core not installed
pytest.importorskip("llama_index.core")

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from fitz_ai.integrations.llamaindex.query_engine import FitzQueryEngine


def create_test_engine(
    mock_optimizer_class,
    cache_hit=False,
    answer="Test answer",
    routing_advice=None,
):
    """Helper to create a test engine with mocked optimizer."""
    mock_optimizer = MagicMock()
    mock_optimizer_class.return_value = mock_optimizer

    mock_lookup_result = MagicMock()
    mock_lookup_result.hit = cache_hit
    mock_lookup_result.answer = answer if cache_hit else None
    mock_lookup_result.sources = [{"source_id": "node1", "excerpt": "text"}] if cache_hit else []
    mock_lookup_result.routing_advice = routing_advice
    mock_optimizer.lookup.return_value = mock_lookup_result

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        NodeWithScore(node=TextNode(text="source", id_="node_1"), score=0.9)
    ]

    mock_llm = MagicMock()
    mock_llm.complete.return_value = answer

    mock_embed_fn = MagicMock(return_value=[1.0] * 1536)

    engine = FitzQueryEngine(
        retriever=mock_retriever,
        llm=mock_llm,
        api_key="fitz_test",
        org_key="a" * 64,
        embedding_fn=mock_embed_fn,
        llm_model="gpt-4o",
    )

    return engine, mock_optimizer, mock_retriever, mock_llm


class TestFitzQueryEngine:
    """Tests for FitzQueryEngine."""

    @patch("fitz_ai.integrations.llamaindex.query_engine.FitzOptimizer")
    def test_extract_chunk_ids(self, mock_optimizer_class):
        """Extracts IDs from nodes."""
        nodes = [
            NodeWithScore(node=TextNode(text="text1", id_="node_1"), score=0.9),
            NodeWithScore(node=TextNode(text="text2", id_="node_2"), score=0.8),
        ]

        engine, _, _, _ = create_test_engine(mock_optimizer_class)
        ids = engine._extract_chunk_ids(nodes)

        assert ids == ["node_1", "node_2"]

    @patch("fitz_ai.integrations.llamaindex.query_engine.FitzOptimizer")
    def test_extract_chunk_ids_fallback(self, mock_optimizer_class):
        """Generates fallback IDs for nodes with empty ID."""
        # Create node with default ID (empty string)
        node1 = TextNode(text="text1", id_="")

        nodes = [NodeWithScore(node=node1, score=0.9)]

        engine, _, _, _ = create_test_engine(mock_optimizer_class)
        ids = engine._extract_chunk_ids(nodes)

        # Should use fallback "node_0" when id is empty
        assert len(ids) == 1
        assert ids[0] == "node_0"

    @patch("fitz_ai.integrations.llamaindex.query_engine.FitzOptimizer")
    def test_node_to_source(self, mock_optimizer_class):
        """Converts node to source dict."""
        node = TextNode(
            text="This is source content for the document.",
            id_="doc_123",
            metadata={"source": "file.pdf", "page": 5},
        )
        node_with_score = NodeWithScore(node=node, score=0.95)

        engine, _, _, _ = create_test_engine(mock_optimizer_class)
        source = engine._node_to_source(node_with_score)

        assert source["source_id"] == "doc_123"
        assert "source content" in source["excerpt"]
        assert source["metadata"]["page"] == 5

    @patch("fitz_ai.integrations.llamaindex.query_engine.FitzOptimizer")
    def test_node_to_source_long_content(self, mock_optimizer_class):
        """Truncates long content in excerpt."""
        long_text = "x" * 1000
        node = TextNode(text=long_text, id_="doc_1")
        node_with_score = NodeWithScore(node=node, score=0.9)

        engine, _, _, _ = create_test_engine(mock_optimizer_class)
        source = engine._node_to_source(node_with_score)

        assert len(source["excerpt"]) <= 500

    @patch("fitz_ai.integrations.llamaindex.query_engine.FitzOptimizer")
    def test_cache_hit_skips_llm(self, mock_optimizer_class):
        """Cache hit returns answer WITHOUT calling LLM."""
        engine, mock_optimizer, mock_retriever, mock_llm = create_test_engine(
            mock_optimizer_class, cache_hit=True, answer="Cached answer"
        )

        query_bundle = QueryBundle(query_str="What is X?")
        response = engine._query(query_bundle)

        # Verify: LLM was NOT called (this is the key assertion!)
        mock_llm.complete.assert_not_called()

        # Verify: cached answer returned
        assert response.response == "Cached answer"
        assert response.metadata["_fitz_cache_hit"] is True

        # Verify: retriever WAS called (we need nodes for fingerprint)
        mock_retriever.retrieve.assert_called_once()

    @patch("fitz_ai.integrations.llamaindex.query_engine.FitzOptimizer")
    def test_cache_miss_runs_llm_and_stores(self, mock_optimizer_class):
        """Cache miss runs LLM and stores result."""
        engine, mock_optimizer, mock_retriever, mock_llm = create_test_engine(
            mock_optimizer_class, cache_hit=False, answer="LLM generated answer"
        )

        query_bundle = QueryBundle(query_str="What is X?")
        response = engine._query(query_bundle)

        # LLM was called
        mock_llm.complete.assert_called_once()

        # Cache was checked
        mock_optimizer.lookup.assert_called_once()

        # Result was stored
        mock_optimizer.store.assert_called_once()
        store_call = mock_optimizer.store.call_args
        assert store_call.kwargs["answer_text"] == "LLM generated answer"

        # Result has correct structure
        assert response.response == "LLM generated answer"
        assert response.metadata["_fitz_cache_hit"] is False

    @patch("fitz_ai.integrations.llamaindex.query_engine.FitzOptimizer")
    def test_query_includes_routing_advice(self, mock_optimizer_class):
        """Query includes routing advice in response metadata."""
        routing = {"complexity": "complex", "recommended_model": "smart"}
        engine, mock_optimizer, _, _ = create_test_engine(
            mock_optimizer_class, cache_hit=False, routing_advice=routing
        )

        query_bundle = QueryBundle(query_str="What is X?")
        response = engine._query(query_bundle)

        # Routing advice stored in engine
        assert engine.get_routing_advice() is not None
        assert engine.get_routing_advice()["complexity"] == "complex"

        # Routing advice in response metadata
        assert response.metadata.get("_fitz_routing") is not None

    @patch("fitz_ai.integrations.llamaindex.query_engine.FitzOptimizer")
    def test_was_cache_hit_property(self, mock_optimizer_class):
        """was_cache_hit property reflects last query."""
        engine, _, _, _ = create_test_engine(
            mock_optimizer_class, cache_hit=True, answer="Cached"
        )

        query_bundle = QueryBundle(query_str="test")
        engine._query(query_bundle)

        assert engine.was_cache_hit is True

    @patch("fitz_ai.integrations.llamaindex.query_engine.FitzOptimizer")
    def test_context_manager(self, mock_optimizer_class):
        """Context manager closes optimizer."""
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer

        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        with FitzQueryEngine(
            retriever=mock_retriever,
            llm=mock_llm,
            api_key="fitz_test",
            org_key="a" * 64,
            embedding_fn=lambda x: [1.0] * 1536,
            llm_model="gpt-4o",
        ) as engine:
            pass

        mock_optimizer.close.assert_called_once()


class TestFitzQueryEngineLegacy:
    """Tests for legacy query_engine parameter (deprecated)."""

    @patch("fitz_ai.integrations.llamaindex.query_engine.FitzOptimizer")
    @patch("fitz_ai.integrations.llamaindex.query_engine.logger")
    def test_legacy_mode_sets_flag(self, mock_logger, mock_optimizer_class):
        """Legacy mode sets _legacy_mode flag and logs warning."""
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer

        mock_query_engine = MagicMock()
        mock_query_engine.callback_manager = None

        engine = FitzQueryEngine(
            retriever=None,
            llm=None,
            query_engine=mock_query_engine,  # Legacy parameter
            api_key="fitz_test",
            org_key="a" * 64,
            embedding_fn=lambda x: [1.0] * 1536,
            llm_model="gpt-4o",
        )

        # In legacy mode
        assert engine._legacy_mode is True

        # Warning was logged
        mock_logger.warning.assert_called_once()
        assert "deprecated" in mock_logger.warning.call_args[0][0].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
