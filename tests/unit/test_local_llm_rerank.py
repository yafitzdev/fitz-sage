# tests/test_local_llm_rerank.py
"""
Tests for local LLM rerank backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestLocalRerankerConfig:
    """Tests for LocalRerankerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from fitz_ai.backends.local_llm.rerank import LocalRerankerConfig

        cfg = LocalRerankerConfig()

        assert cfg.top_k == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        from fitz_ai.backends.local_llm.rerank import LocalRerankerConfig

        cfg = LocalRerankerConfig(top_k=5)

        assert cfg.top_k == 5

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        from fitz_ai.backends.local_llm.rerank import LocalRerankerConfig

        cfg = LocalRerankerConfig()

        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.top_k = 20


class TestCosineFunction:
    """Tests for _cosine helper function."""

    def test_cosine_identical_vectors(self):
        """Test cosine similarity of identical normalized vectors."""
        from fitz_ai.backends.local_llm.rerank import _cosine

        # Normalized unit vector
        v = [1.0, 0.0, 0.0]
        result = _cosine(v, v)

        assert result == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        from fitz_ai.backends.local_llm.rerank import _cosine

        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        result = _cosine(v1, v2)

        assert result == pytest.approx(0.0)

    def test_cosine_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        from fitz_ai.backends.local_llm.rerank import _cosine

        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        result = _cosine(v1, v2)

        assert result == pytest.approx(-1.0)

    def test_cosine_similar_vectors(self):
        """Test cosine similarity of similar vectors."""
        from fitz_ai.backends.local_llm.rerank import _cosine

        v1 = [0.8, 0.6, 0.0]
        v2 = [0.7, 0.7, 0.14]
        result = _cosine(v1, v2)

        # Should be positive and high
        assert result > 0.9

    def test_cosine_different_lengths(self):
        """Test cosine handles different length vectors."""
        from fitz_ai.backends.local_llm.rerank import _cosine

        v1 = [1.0, 0.5, 0.3]
        v2 = [1.0, 0.5]  # Shorter
        result = _cosine(v1, v2)

        # Should only compute for min length (2 elements)
        # 1.0*1.0 + 0.5*0.5 = 1.25
        assert result == pytest.approx(1.25)

    def test_cosine_empty_vectors(self):
        """Test cosine with empty vectors."""
        from fitz_ai.backends.local_llm.rerank import _cosine

        result = _cosine([], [])

        assert result == 0.0


class TestLocalReranker:
    """Tests for LocalReranker."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        from fitz_ai.backends.local_llm.rerank import LocalReranker

        mock_embedder = MagicMock()
        reranker = LocalReranker(mock_embedder)

        assert reranker._emb is mock_embedder
        assert reranker._cfg.top_k == 10

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        from fitz_ai.backends.local_llm.rerank import LocalReranker, LocalRerankerConfig

        mock_embedder = MagicMock()
        cfg = LocalRerankerConfig(top_k=3)
        reranker = LocalReranker(mock_embedder, cfg)

        assert reranker._cfg.top_k == 3

    def test_rerank_empty_candidates(self):
        """Test rerank with empty candidates returns empty."""
        from fitz_ai.backends.local_llm.rerank import LocalReranker

        mock_embedder = MagicMock()
        reranker = LocalReranker(mock_embedder)

        result = reranker.rerank("query", [])

        assert result == []

    def test_rerank_returns_sorted_by_score(self):
        """Test rerank returns candidates sorted by similarity score."""
        from fitz_ai.backends.local_llm.rerank import LocalReranker

        mock_embedder = MagicMock()
        # Query embedding
        query_emb = [1.0, 0.0, 0.0]
        # Candidate embeddings - doc2 most similar, then doc0, then doc1
        doc_embs = [
            [0.8, 0.6, 0.0],  # doc0: somewhat similar
            [0.0, 1.0, 0.0],  # doc1: orthogonal
            [0.95, 0.31, 0.0],  # doc2: most similar
        ]

        mock_embedder.embed_texts.side_effect = [
            [query_emb],  # First call for query
            doc_embs,  # Second call for candidates
        ]

        reranker = LocalReranker(mock_embedder)
        result = reranker.rerank("query", ["doc0", "doc1", "doc2"])

        # Should be sorted descending by score
        assert len(result) == 3
        assert result[0][0] == 2  # doc2 index (most similar)
        assert result[1][0] == 0  # doc0 index
        assert result[2][0] == 1  # doc1 index (least similar)

        # Scores should be descending
        assert result[0][1] > result[1][1] > result[2][1]

    def test_rerank_respects_top_k(self):
        """Test rerank limits results to top_k."""
        from fitz_ai.backends.local_llm.rerank import LocalReranker, LocalRerankerConfig

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.side_effect = [
            [[1.0, 0.0]],  # Query
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],  # 5 docs
        ]

        cfg = LocalRerankerConfig(top_k=2)
        reranker = LocalReranker(mock_embedder, cfg)

        result = reranker.rerank("query", ["d0", "d1", "d2", "d3", "d4"])

        assert len(result) == 2

    def test_rerank_calls_embedder(self):
        """Test rerank calls embedder.embed_texts correctly."""
        from fitz_ai.backends.local_llm.rerank import LocalReranker

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.side_effect = [
            [[0.1, 0.2]],  # Query
            [[0.3, 0.4], [0.5, 0.6]],  # Docs
        ]

        reranker = LocalReranker(mock_embedder)
        reranker.rerank("the query", ["doc A", "doc B"])

        # Should call embed_texts twice: once for query, once for candidates
        assert mock_embedder.embed_texts.call_count == 2
        mock_embedder.embed_texts.assert_any_call(["the query"])
        mock_embedder.embed_texts.assert_any_call(["doc A", "doc B"])

    def test_rerank_single_candidate(self):
        """Test rerank with single candidate."""
        from fitz_ai.backends.local_llm.rerank import LocalReranker

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.side_effect = [
            [[1.0, 0.0]],  # Query
            [[0.8, 0.6]],  # Single doc
        ]

        reranker = LocalReranker(mock_embedder)
        result = reranker.rerank("query", ["only doc"])

        assert len(result) == 1
        assert result[0][0] == 0  # Index
        assert result[0][1] > 0  # Positive score

    def test_rerank_returns_tuples_of_index_score(self):
        """Test rerank returns list of (index, score) tuples."""
        from fitz_ai.backends.local_llm.rerank import LocalReranker

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.side_effect = [
            [[1.0]],  # Query
            [[0.5], [0.8]],  # Docs
        ]

        reranker = LocalReranker(mock_embedder)
        result = reranker.rerank("q", ["a", "b"])

        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], int)  # index
            assert isinstance(item[1], float)  # score
