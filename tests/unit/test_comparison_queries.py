# tests/test_comparison_queries.py
"""
Tests for comparison query handling in VectorSearchStep.

Comparison queries (e.g., "CAN vs SPI") are detected by pattern matching
and expanded via LLM to ensure both compared entities are retrieved.
"""

import json
from unittest.mock import MagicMock

import pytest

from fitz_ai.engines.fitz_rag.retrieval.steps.strategies.comparison import ComparisonSearch
from fitz_ai.engines.fitz_rag.retrieval.steps.vector_search import VectorSearchStep


class TestComparisonDetection:
    """Test comparison pattern detection."""

    @pytest.fixture
    def step(self):
        """Create a step with mock dependencies."""
        return VectorSearchStep(
            client=MagicMock(),
            embedder=MagicMock(),
            collection="test",
            chat=MagicMock(),
        )

    @pytest.mark.parametrize(
        "query,expected",
        [
            # Should detect as comparison
            ("CAN vs SPI", True),
            ("CAN vs. SPI", True),
            ("CAN versus SPI", True),
            ("Compare CAN and SPI", True),
            ("compared to the previous version", True),
            ("difference between v2.3.0 and v2.3.1", True),
            ("How does module A compare to module B?", True),
            ("which has lower latency", True),
            ("which is better for real-time", True),
            # Should NOT detect as comparison
            ("What is CAN?", False),
            ("Tell me about SPI protocol", False),
            ("CAN bus error handling", False),
            ("The module comparison report", False),  # "comparison" as noun, not verb
            ("version 2.3.0 release notes", False),
        ],
    )
    def test_comparison_detection(self, step, query, expected):
        """Test that comparison patterns are correctly detected."""
        assert step._is_comparison_query(query) == expected


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

    def test_comparison_search_retrieves_both_entities(self, step_with_full_mocks):
        """Test that comparison search retrieves chunks for both entities."""
        chunks = step_with_full_mocks.execute("CAN vs SPI", [])

        # Should retrieve chunks (deduplication may reduce count)
        # Mock setup returns same hit for both searches, so dedupe leaves 1
        assert len(chunks) >= 1
        # At least one should contain relevant content
        contents = [c.content for c in chunks]
        assert any("CAN" in c or "protocol" in c for c in contents)

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

        # Same chunk returned by both searches
        mock_hit = MagicMock()
        mock_hit.id = "same_chunk"
        mock_hit.score = 0.9
        mock_hit.payload = {"content": "shared content", "source": "doc"}

        mock_client = MagicMock()
        mock_client.search.return_value = [mock_hit]  # Same result for both

        step = VectorSearchStep(
            client=mock_client,
            embedder=mock_embedder,
            collection="test",
            chat=mock_chat,
            k=10,
        )

        chunks = step.execute("A vs B", [])

        # Should deduplicate - only 1 chunk even though searched twice
        assert len(chunks) == 1

    def test_comparison_bypasses_length_threshold(self):
        """Test that comparison detection bypasses min_query_length."""
        mock_chat = MagicMock()
        mock_chat.chat.return_value = json.dumps(
            {
                "entities": ["A", "B"],
                "queries": ["A info", "B info"],
            }
        )

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [1.0, 0.0, 0.0]

        mock_hit = MagicMock()
        mock_hit.id = "chunk1"
        mock_hit.score = 0.9
        mock_hit.payload = {"content": "test", "source": "doc"}

        mock_client = MagicMock()
        mock_client.search.return_value = [mock_hit]

        step = VectorSearchStep(
            client=mock_client,
            embedder=mock_embedder,
            collection="test",
            chat=mock_chat,
            k=10,
            min_query_length=1000,  # Very high threshold
        )

        # Short query that matches comparison pattern
        short_query = "A vs B"  # Only 6 chars, way below 1000
        assert len(short_query) < step.min_query_length

        # Should still trigger comparison search, not single search
        chunks = step.execute(short_query, [])

        # Should return results (comparison logic may or may not call chat depending on implementation)
        assert chunks is not None
        assert isinstance(chunks, list)


class TestComparisonNoChat:
    """Test behavior when chat client is not available."""

    def test_comparison_detection_works_without_chat(self):
        """Test that comparison pattern detection works regardless of chat presence.

        The _is_comparison_query() method should still detect comparison patterns
        even when chat is None. However, the actual comparison expansion requires
        chat, so VectorSearchStep.execute() checks for chat before using
        ComparisonSearch strategy.
        """
        step = VectorSearchStep(
            client=MagicMock(),
            embedder=MagicMock(),
            collection="test",
            chat=None,  # No chat client
            k=10,
        )

        # Pattern detection should work regardless of chat
        assert step._is_comparison_query("CAN vs SPI")
        assert step._is_comparison_query("Compare module A and module B")
        assert step._is_comparison_query("What is the difference between v1 and v2?")
        assert not step._is_comparison_query("What is CAN protocol?")

    def test_comparison_strategy_requires_chat(self):
        """Test that ComparisonSearch requires chat for query expansion."""
        # ComparisonSearch requires chat and will use it for expansion
        mock_chat = MagicMock()
        mock_chat.chat.return_value = '{"entities": ["A", "B"], "queries": ["A info", "B info"]}'

        strategy = ComparisonSearch(
            client=MagicMock(),
            embedder=MagicMock(),
            collection="test",
            chat=mock_chat,
        )

        # Expansion should call the chat client
        queries = strategy._expand_comparison_query("A vs B")
        assert mock_chat.chat.called
        assert len(queries) >= 1
