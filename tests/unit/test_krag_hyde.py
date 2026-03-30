# tests/unit/test_krag_hyde.py
"""
Unit tests for HyDE (Hypothetical Document Embeddings) in KRAG search strategies.

HyDE is owned by the retrieval router — it generates hypotheses, embeds them,
and passes pre-computed vectors to strategies. Strategies never call the HyDE
generator directly; they only consume hyde_vectors.

Tests cover:
  - Strategies use pre-computed HyDE vectors for additional searches
  - HyDE results merged with 0.5 weight discount
  - Strategies work correctly when no HyDE vectors are provided
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fitz_sage.engines.fitz_krag.retrieval.strategies.code_search import CodeSearchStrategy
from fitz_sage.engines.fitz_krag.retrieval.strategies.section_search import (
    SectionSearchStrategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> MagicMock:
    """Create a mock FitzKragConfig with fields strategies read."""
    cfg = MagicMock()
    cfg.keyword_weight = overrides.get("keyword_weight", 0.4)
    cfg.semantic_weight = overrides.get("semantic_weight", 0.6)
    cfg.code_bm25_weight = overrides.get("code_bm25_weight", 0.3)
    cfg.section_bm25_weight = overrides.get("section_bm25_weight", 0.6)
    cfg.section_semantic_weight = overrides.get("section_semantic_weight", 0.4)
    return cfg


def _make_symbol_row(row_id: str, score: float = 0.8, name: str = "func") -> dict:
    """Build a dict mimicking a symbol store row."""
    return {
        "id": row_id,
        "raw_file_id": f"raw_{row_id}",
        "qualified_name": f"mod.{name}",
        "name": name,
        "kind": "function",
        "start_line": 1,
        "end_line": 10,
        "score": score,
        "summary": f"Summary for {name}",
    }


def _make_section_row(row_id: str, score: float = 0.7, title: str = "Setup") -> dict:
    """Build a dict mimicking a section store row."""
    return {
        "id": row_id,
        "raw_file_id": f"raw_{row_id}",
        "title": title,
        "level": 2,
        "score": score,
        "summary": f"Section about {title}",
    }


# ---------------------------------------------------------------------------
# TestCodeSearchHyDE
# ---------------------------------------------------------------------------


class TestCodeSearchHyDE:
    """Tests for HyDE vector consumption in CodeSearchStrategy."""

    def test_hyde_vectors_used_for_additional_searches(self):
        """Pre-computed HyDE vectors trigger extra vector searches."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        kw_row = _make_symbol_row("kw1", score=0.9, name="authenticate")
        store.search_by_name.return_value = [kw_row]
        store.search_bm25.return_value = []

        sem_row = _make_symbol_row("sem1", score=0.85, name="login")
        hyde_row = _make_symbol_row("hyde1", score=0.7, name="verify_token")

        embedder.embed.return_value = [0.1] * 3
        store.search_by_vector.side_effect = [
            [sem_row],  # semantic search
            [hyde_row],  # HyDE vector 1
            [_make_symbol_row("hyde2", score=0.6, name="check_session")],  # HyDE vector 2
        ]

        strategy = CodeSearchStrategy(store, embedder, config)

        # Pass pre-computed HyDE vectors (as router would)
        hyde_vectors = [[0.2] * 3, [0.3] * 3]
        result = strategy.retrieve("how does auth work", limit=5, hyde_vectors=hyde_vectors)

        # Vector search called 3 times: 1 semantic + 2 HyDE vectors
        assert store.search_by_vector.call_count == 3
        assert len(result) > 0

    def test_hyde_results_merged_with_lower_weight(self):
        """HyDE results have scores discounted by 0.5 during merge."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        strategy = CodeSearchStrategy(store, embedder, config)

        semantic_results = [_make_symbol_row("sem1", score=0.8)]
        hyde_results = [
            {
                "id": "hyde1",
                "score": 0.9,
                "raw_file_id": "f",
                "qualified_name": "q",
                "name": "n",
                "kind": "function",
                "start_line": 1,
                "end_line": 5,
            }
        ]

        merged = strategy._merge_hyde(semantic_results, hyde_results)

        hyde_in_merged = [r for r in merged if r["id"] == "hyde1"]
        assert len(hyde_in_merged) == 1
        assert hyde_in_merged[0]["score"] == pytest.approx(0.45)

        sem_in_merged = [r for r in merged if r["id"] == "sem1"]
        assert len(sem_in_merged) == 1
        assert sem_in_merged[0]["score"] == 0.8

    def test_search_works_without_hyde_vectors(self):
        """Without HyDE vectors, strategy uses keyword + BM25 + semantic only."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        kw_row = _make_symbol_row("kw1", score=0.9)
        store.search_by_name.return_value = [kw_row]
        store.search_bm25.return_value = []

        sem_row = _make_symbol_row("sem1", score=0.8)
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [sem_row]

        strategy = CodeSearchStrategy(store, embedder, config)

        result = strategy.retrieve("find func", limit=5)

        embedder.embed.assert_called_once_with("find func", task_type="query")
        store.search_by_vector.assert_called_once()
        assert len(result) > 0

    def test_empty_hyde_vectors_skips_hyde(self):
        """Empty hyde_vectors list (router skipped HyDE) means no HyDE search."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        store.search_by_name.return_value = [_make_symbol_row("kw1")]
        store.search_bm25.return_value = []
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [_make_symbol_row("sem1")]

        strategy = CodeSearchStrategy(store, embedder, config)
        result = strategy.retrieve("find func", limit=5, hyde_vectors=[])

        # Only 1 vector search (semantic), no HyDE searches
        store.search_by_vector.assert_called_once()
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestSectionSearchHyDE
# ---------------------------------------------------------------------------


class TestSectionSearchHyDE:
    """Tests for HyDE vector consumption in SectionSearchStrategy."""

    def test_hyde_vectors_used_for_additional_searches(self):
        """Pre-computed HyDE vectors trigger extra vector searches."""
        store = MagicMock(name="section_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        bm25_row = _make_section_row("bm1", score=0.8, title="Installation")
        store.search_bm25.return_value = [bm25_row]

        sem_row = _make_section_row("sem1", score=0.7, title="Configuration")
        hyde_row = _make_section_row("hyde1", score=0.6, title="Deployment")

        embedder.embed.return_value = [0.1]
        store.search_by_vector.side_effect = [
            [sem_row],  # semantic
            [hyde_row],  # HyDE vector 1
        ]

        strategy = SectionSearchStrategy(store, embedder, config)

        hyde_vectors = [[0.2]]
        result = strategy.retrieve("how to deploy", limit=5, hyde_vectors=hyde_vectors)

        assert store.search_by_vector.call_count == 2  # 1 semantic + 1 HyDE
        assert len(result) > 0

    def test_hyde_results_merged_with_lower_weight(self):
        """HyDE results are discounted by 0.5 in SectionSearchStrategy."""
        store = MagicMock(name="section_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        strategy = SectionSearchStrategy(store, embedder, config)

        semantic_results = [_make_section_row("sem1", score=0.8)]
        hyde_results = [{"id": "h1", "score": 1.0, "raw_file_id": "f", "title": "T", "level": 2}]

        merged = strategy._merge_hyde(semantic_results, hyde_results)

        hyde_merged = [r for r in merged if r["id"] == "h1"]
        assert len(hyde_merged) == 1
        assert hyde_merged[0]["score"] == pytest.approx(0.5)

    def test_search_works_without_hyde_vectors(self):
        """Without HyDE vectors, section strategy uses BM25 + semantic only."""
        store = MagicMock(name="section_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        store.search_bm25.return_value = [_make_section_row("bm1")]
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [_make_section_row("sem1")]

        strategy = SectionSearchStrategy(store, embedder, config)

        result = strategy.retrieve("setup guide", limit=5)

        embedder.embed.assert_called_once_with("setup guide", task_type="query")
        store.search_by_vector.assert_called_once()
        assert len(result) > 0

    def test_empty_hyde_vectors_skips_hyde(self):
        """Empty hyde_vectors list (router skipped HyDE) means no HyDE search."""
        store = MagicMock(name="section_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        store.search_bm25.return_value = [_make_section_row("bm1")]
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [_make_section_row("sem1")]

        strategy = SectionSearchStrategy(store, embedder, config)
        result = strategy.retrieve("setup guide", limit=5, hyde_vectors=[])

        store.search_by_vector.assert_called_once()
        assert len(result) > 0
