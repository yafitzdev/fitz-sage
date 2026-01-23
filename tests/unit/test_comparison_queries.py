# tests/unit/test_comparison_queries.py
"""
Tests for comparison query handling.

Comparison queries are detected by LLM classification and expanded
via LLM to ensure both compared entities are retrieved.
"""

import json
from unittest.mock import MagicMock

import pytest

from fitz_ai.engines.fitz_rag.retrieval.steps.strategies.comparison import ComparisonSearch
from fitz_ai.engines.fitz_rag.retrieval.steps.vector_search import VectorSearchStep
from fitz_ai.retrieval.detection import DetectionOrchestrator


class TestComparisonDetection:
    """Test comparison detection via LLM classification."""

    @pytest.fixture
    def mock_chat(self):
        """Create a mock chat client that returns comparison detection."""
        mock = MagicMock()
        return mock

    def test_comparison_detected_with_llm(self, mock_chat):
        """Test that LLM classification detects comparison queries."""
        # LLM returns classification with comparison detected
        mock_chat.chat.return_value = json.dumps(
            {
                "temporal": {"detected": False},
                "aggregation": {"detected": False},
                "comparison": {
                    "detected": True,
                    "entities": ["CAN", "SPI"],
                    "comparison_queries": ["CAN protocol", "SPI protocol", "CAN vs SPI"],
                },
                "freshness": {"boost_recency": False, "boost_authority": False},
                "rewriter": {"needs_context": False, "is_compound": False},
            }
        )

        orchestrator = DetectionOrchestrator(chat_client=mock_chat)
        summary = orchestrator.detect_for_retrieval("CAN vs SPI")

        assert summary.has_comparison_intent
        assert summary.comparison_entities == ["CAN", "SPI"]
        assert "CAN protocol" in summary.comparison_queries

    def test_comparison_not_detected_for_regular_query(self, mock_chat):
        """Test that regular queries don't trigger comparison detection."""
        mock_chat.chat.return_value = json.dumps(
            {
                "temporal": {"detected": False},
                "aggregation": {"detected": False},
                "comparison": {"detected": False, "entities": [], "comparison_queries": []},
                "freshness": {"boost_recency": False, "boost_authority": False},
                "rewriter": {"needs_context": False, "is_compound": False},
            }
        )

        orchestrator = DetectionOrchestrator(chat_client=mock_chat)
        summary = orchestrator.detect_for_retrieval("What is CAN?")

        assert not summary.has_comparison_intent

    def test_detection_without_chat_client(self):
        """Test that detection without chat client skips LLM classification."""
        orchestrator = DetectionOrchestrator(chat_client=None)
        summary = orchestrator.detect_for_retrieval("CAN vs SPI")

        # Without chat client, no LLM classification happens
        assert not summary.has_comparison_intent
        assert not summary.has_temporal_intent
        assert not summary.has_aggregation_intent


class TestComparisonExpansion:
    """Test comparison query expansion via LLM."""

    @pytest.fixture
    def strategy_with_mock_chat(self):
        """Create a comparison strategy with a mock chat that returns valid JSON."""
        mock_chat = MagicMock()
        mock_chat.chat.return_value = json.dumps(
            {
                "entities": ["CAN", "SPI"],
                "queries": [
                    "CAN protocol latency",
                    "CAN bus performance",
                    "SPI protocol latency",
                    "SPI bus performance",
                    "CAN vs SPI comparison",
                ],
            }
        )
        return ComparisonSearch(
            client=MagicMock(),
            embedder=MagicMock(),
            collection="test",
            chat=mock_chat,
        )

    def test_comparison_expansion_returns_queries(self, strategy_with_mock_chat):
        """Test that comparison expansion returns multiple queries."""
        queries = strategy_with_mock_chat._expand_comparison_query(
            "CAN vs SPI - which has lower latency?"
        )

        assert len(queries) >= 4  # At least 2 per entity
        assert any("CAN" in q for q in queries)
        assert any("SPI" in q for q in queries)

    def test_comparison_expansion_calls_chat(self, strategy_with_mock_chat):
        """Test that comparison expansion calls the chat client."""
        strategy_with_mock_chat._expand_comparison_query("CAN vs SPI")

        strategy_with_mock_chat.chat.chat.assert_called_once()
        call_args = strategy_with_mock_chat.chat.chat.call_args[0][0]
        assert len(call_args) == 1
        assert "comparison query" in call_args[0]["content"].lower()

    def test_comparison_expansion_handles_code_blocks(self):
        """Test that comparison expansion handles markdown code blocks."""
        mock_chat = MagicMock()
        mock_chat.chat.return_value = """```json
{
    "entities": ["v2.3.0", "v2.3.1"],
    "queries": ["v2.3.0 features", "v2.3.1 changes", "v2.3.0 vs v2.3.1"]
}
```"""
        strategy = ComparisonSearch(
            client=MagicMock(),
            embedder=MagicMock(),
            collection="test",
            chat=mock_chat,
        )

        queries = strategy._expand_comparison_query("difference between v2.3.0 and v2.3.1")

        assert len(queries) == 3
        assert any("v2.3.0" in q for q in queries)
        assert any("v2.3.1" in q for q in queries)


class TestComparisonFallback:
    """Test fallback behavior when comparison parsing fails."""

    def test_fallback_on_invalid_json(self):
        """Test that invalid JSON falls back to generic expansion."""
        mock_chat = MagicMock()
        # First call returns invalid JSON (for comparison expansion)
        # Second call returns valid JSON array (for generic expansion fallback)
        mock_chat.chat.side_effect = [
            "this is not valid json",
            '["query 1", "query 2", "query 3"]',
        ]

        strategy = ComparisonSearch(
            client=MagicMock(),
            embedder=MagicMock(),
            collection="test",
            chat=mock_chat,
        )

        queries = strategy._expand_comparison_query("CAN vs SPI")

        # Should have called chat twice (comparison + fallback)
        assert mock_chat.chat.call_count == 2
        assert isinstance(queries, list)

    def test_fallback_on_missing_queries_key(self):
        """Test that missing 'queries' key falls back to generic expansion."""
        mock_chat = MagicMock()
        # First call returns JSON without 'queries' key
        # Second call returns valid JSON array for fallback
        mock_chat.chat.side_effect = [
            '{"entities": ["A", "B"]}',  # Missing 'queries' key
            '["fallback query 1", "fallback query 2"]',
        ]

        strategy = ComparisonSearch(
            client=MagicMock(),
            embedder=MagicMock(),
            collection="test",
            chat=mock_chat,
        )

        queries = strategy._expand_comparison_query("A vs B")

        assert mock_chat.chat.call_count == 2
        assert isinstance(queries, list)


class TestComparisonSearchIntegration:
    """Integration tests for comparison search flow."""

    @pytest.fixture
    def step_with_full_mocks(self):
        """Create a step with all mocks configured."""
        mock_chat = MagicMock()
        mock_chat.chat.return_value = json.dumps(
            {
                "entities": ["CAN", "SPI"],
                "queries": ["CAN protocol", "SPI protocol"],
            }
        )

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [1.0, 0.0, 0.0]

        # Create mock hits
        mock_hit_1 = MagicMock()
        mock_hit_1.id = "chunk1"
        mock_hit_1.score = 0.9
        mock_hit_1.payload = {"content": "CAN protocol info", "source": "doc1"}

        mock_hit_2 = MagicMock()
        mock_hit_2.id = "chunk2"
        mock_hit_2.score = 0.85
        mock_hit_2.payload = {"content": "SPI protocol info", "source": "doc2"}

        mock_client = MagicMock()
        # Return different results for different searches
        mock_client.search.side_effect = [
            [mock_hit_1],  # CAN query
            [mock_hit_2],  # SPI query
        ]

        return VectorSearchStep(
            client=mock_client,
            embedder=mock_embedder,
            collection="test",
            chat=mock_chat,
            k=10,
        )

    def test_comparison_search_deduplicates_results(self):
        """Test that comparison search deduplicates results across queries."""
        mock_chat = MagicMock()
        mock_chat.chat.return_value = json.dumps(
            {
                "entities": ["A", "B"],
                "queries": ["query A", "query B"],
            }
        )

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [1.0, 0.0, 0.0]
        mock_embedder.embed_batch.return_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        # Same chunk returned by both searches
        mock_hit = MagicMock()
        mock_hit.id = "same_chunk"
        mock_hit.score = 0.9
        mock_hit.payload = {"content": "shared content", "source": "doc"}

        mock_client = MagicMock()
        mock_client.search.return_value = [mock_hit]  # Same result for both

        strategy = ComparisonSearch(
            client=mock_client,
            embedder=mock_embedder,
            collection="test",
            chat=mock_chat,
            k=10,
        )

        # Execute comparison search
        chunks = strategy.execute("A vs B", [])

        # Should deduplicate - only 1 chunk even though searched twice
        assert len(chunks) == 1


class TestLLMClassifier:
    """Test the LLM classifier directly."""

    def test_classifier_parses_json_response(self):
        """Test that classifier parses valid JSON responses."""
        from fitz_ai.retrieval.detection import DetectionCategory
        from fitz_ai.retrieval.detection.llm_classifier import LLMClassifier

        mock_chat = MagicMock()
        mock_chat.chat.return_value = json.dumps(
            {
                "temporal": {"detected": True, "intent": "COMPARISON"},
                "aggregation": {"detected": False},
                "comparison": {"detected": True, "entities": ["A", "B"]},
                "freshness": {"boost_recency": True, "boost_authority": False},
                "rewriter": {"needs_context": False, "is_compound": False},
            }
        )

        classifier = LLMClassifier(chat_client=mock_chat)
        result = classifier.classify("A vs B from last month")

        # Result is keyed by DetectionCategory enum
        assert result[DetectionCategory.TEMPORAL].detected is True
        assert result[DetectionCategory.COMPARISON].detected is True
        assert result[DetectionCategory.FRESHNESS].metadata["boost_recency"] is True

    def test_classifier_handles_markdown_code_blocks(self):
        """Test that classifier extracts JSON from markdown code blocks."""
        from fitz_ai.retrieval.detection import DetectionCategory
        from fitz_ai.retrieval.detection.llm_classifier import LLMClassifier

        mock_chat = MagicMock()
        mock_chat.chat.return_value = """Here's the classification:

```json
{
    "temporal": {"detected": false},
    "aggregation": {"detected": true, "type": "LIST"},
    "comparison": {"detected": false},
    "freshness": {"boost_recency": false, "boost_authority": false},
    "rewriter": {"needs_context": false, "is_compound": false}
}
```

The query is asking for a list."""

        classifier = LLMClassifier(chat_client=mock_chat)
        result = classifier.classify("list all test cases")

        # Result is keyed by DetectionCategory enum
        assert result[DetectionCategory.AGGREGATION].detected is True
        # intent contains the parsed AggregationType enum
        from fitz_ai.retrieval.detection import AggregationType

        assert result[DetectionCategory.AGGREGATION].intent == AggregationType.LIST

    def test_classifier_returns_empty_on_failure(self):
        """Test that classifier returns empty result on parse failure."""
        from fitz_ai.retrieval.detection import DetectionCategory
        from fitz_ai.retrieval.detection.llm_classifier import LLMClassifier

        mock_chat = MagicMock()
        mock_chat.chat.return_value = "This is not valid JSON at all"

        classifier = LLMClassifier(chat_client=mock_chat)
        result = classifier.classify("some query")

        # Should return not-detected for all categories
        assert result[DetectionCategory.TEMPORAL].detected is False
        assert result[DetectionCategory.AGGREGATION].detected is False
        assert result[DetectionCategory.COMPARISON].detected is False
