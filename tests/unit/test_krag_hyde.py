# tests/unit/test_krag_hyde.py
"""
Unit tests for HyDE (Hypothetical Document Embeddings) in KRAG search strategies.

Tests HyDE integration in CodeSearchStrategy and SectionSearchStrategy:
hypothesis generation, embedding, merging with 0.5 weight discount, and
graceful fallback when HyDE is disabled or fails.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fitz_ai.engines.fitz_krag.retrieval.strategies.code_search import CodeSearchStrategy
from fitz_ai.engines.fitz_krag.retrieval.strategies.section_search import (
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
    """Tests for HyDE in CodeSearchStrategy."""

    def test_hyde_generates_hypotheses_and_searches(self):
        """HyDE generator produces hypotheses; each is embedded and searched."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        # Keyword returns one result
        kw_row = _make_symbol_row("kw1", score=0.9, name="authenticate")
        store.search_by_name.return_value = [kw_row]

        # BM25 returns nothing
        store.search_bm25.return_value = []

        # Semantic returns one result
        sem_row = _make_symbol_row("sem1", score=0.85, name="login")
        embedder.embed.side_effect = lambda text: [0.1] * 3  # dummy vector
        store.search_by_vector.return_value = [sem_row]

        # HyDE generator
        hyde = MagicMock(name="hyde_generator")
        hyde.generate.return_value = ["def authenticate(user): ...", "def login(creds): ..."]

        # HyDE vector search returns new results
        hyde_row = _make_symbol_row("hyde1", score=0.7, name="verify_token")
        # search_by_vector: 1st call = semantic, then 2 HyDE calls
        store.search_by_vector.side_effect = [
            [sem_row],
            [hyde_row],
            [_make_symbol_row("hyde2", score=0.6, name="check_session")],
        ]

        strategy = CodeSearchStrategy(store, embedder, config)
        strategy._hyde_generator = hyde

        result = strategy.retrieve("how does auth work", limit=5)

        # HyDE generator called
        hyde.generate.assert_called_once_with("how does auth work")

        # Embedder called 3 times: 1 for query, 2 for hypotheses
        assert embedder.embed.call_count == 3

        # Vector search called 3 times: 1 semantic + 2 HyDE hypotheses
        assert store.search_by_vector.call_count == 3

        # Results are addresses
        assert len(result) > 0

    def test_hyde_results_merged_with_lower_weight(self):
        """HyDE results have scores discounted by 0.5 during merge."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        store.search_by_name.return_value = []
        store.search_bm25.return_value = []

        sem_row = _make_symbol_row("sem1", score=0.8)
        hyde_row = _make_symbol_row("hyde1", score=0.9, name="from_hyde")

        embedder.embed.return_value = [0.1]
        store.search_by_vector.side_effect = [
            [sem_row],  # semantic search
            [hyde_row],  # HyDE search for hypothesis
        ]

        hyde = MagicMock(name="hyde_generator")
        hyde.generate.return_value = ["hypothesis 1"]

        strategy = CodeSearchStrategy(store, embedder, config)
        strategy._hyde_generator = hyde

        # Call _merge_hyde directly to verify weight discount
        semantic_results = [sem_row]
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

        # HyDE score should be 0.9 * 0.5 = 0.45
        hyde_in_merged = [r for r in merged if r["id"] == "hyde1"]
        assert len(hyde_in_merged) == 1
        assert hyde_in_merged[0]["score"] == pytest.approx(0.45)

        # Original semantic score unchanged
        sem_in_merged = [r for r in merged if r["id"] == "sem1"]
        assert len(sem_in_merged) == 1
        assert sem_in_merged[0]["score"] == 0.8

    def test_search_works_when_hyde_generator_is_none(self):
        """Without HyDE, strategy uses keyword + BM25 + semantic only."""
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
        assert strategy._hyde_generator is None

        result = strategy.retrieve("find func", limit=5)

        # No HyDE calls: embedder called once (query), vector search once
        embedder.embed.assert_called_once_with("find func")
        store.search_by_vector.assert_called_once()

        assert len(result) > 0

    def test_hyde_failure_does_not_break_search(self):
        """When HyDE generator raises, search proceeds with keyword + semantic."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        kw_row = _make_symbol_row("kw1", score=0.9)
        store.search_by_name.return_value = [kw_row]
        store.search_bm25.return_value = []

        sem_row = _make_symbol_row("sem1", score=0.8)
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [sem_row]

        hyde = MagicMock(name="hyde_generator")
        hyde.generate.side_effect = RuntimeError("HyDE model unavailable")

        strategy = CodeSearchStrategy(store, embedder, config)
        strategy._hyde_generator = hyde

        result = strategy.retrieve("find func", limit=5)

        # HyDE was attempted
        hyde.generate.assert_called_once()

        # Search still returns results from keyword + semantic
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestSectionSearchHyDE
# ---------------------------------------------------------------------------


class TestSectionSearchHyDE:
    """Tests for HyDE in SectionSearchStrategy."""

    def test_hyde_generates_hypotheses_and_searches(self):
        """HyDE generator produces hypotheses; each is embedded and searched."""
        store = MagicMock(name="section_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        bm25_row = _make_section_row("bm1", score=0.8, title="Installation")
        store.search_bm25.return_value = [bm25_row]

        sem_row = _make_section_row("sem1", score=0.7, title="Configuration")
        hyde_row = _make_section_row("hyde1", score=0.6, title="Deployment")

        embedder.embed.side_effect = lambda text: [0.1]
        store.search_by_vector.side_effect = [
            [sem_row],  # semantic
            [hyde_row],  # HyDE hypothesis 1
        ]

        hyde = MagicMock(name="hyde_generator")
        hyde.generate.return_value = ["## How to deploy the app"]

        strategy = SectionSearchStrategy(store, embedder, config)
        strategy._hyde_generator = hyde

        result = strategy.retrieve("how to deploy", limit=5)

        hyde.generate.assert_called_once_with("how to deploy")
        assert embedder.embed.call_count == 2  # 1 query + 1 hypothesis
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

    def test_search_works_when_hyde_generator_is_none(self):
        """Without HyDE, section strategy uses BM25 + semantic only."""
        store = MagicMock(name="section_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        store.search_bm25.return_value = [_make_section_row("bm1")]
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [_make_section_row("sem1")]

        strategy = SectionSearchStrategy(store, embedder, config)
        assert strategy._hyde_generator is None

        result = strategy.retrieve("setup guide", limit=5)

        embedder.embed.assert_called_once_with("setup guide")
        store.search_by_vector.assert_called_once()
        assert len(result) > 0

    def test_hyde_failure_does_not_break_section_search(self):
        """When HyDE raises in section search, BM25 + semantic proceed normally."""
        store = MagicMock(name="section_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        store.search_bm25.return_value = [_make_section_row("bm1")]
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [_make_section_row("sem1")]

        hyde = MagicMock(name="hyde_generator")
        hyde.generate.side_effect = RuntimeError("HyDE timeout")

        strategy = SectionSearchStrategy(store, embedder, config)
        strategy._hyde_generator = hyde

        result = strategy.retrieve("setup guide", limit=5)

        hyde.generate.assert_called_once()
        assert len(result) > 0
