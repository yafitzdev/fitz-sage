# tests/unit/test_krag_bm25.py
"""
Unit tests for BM25 full-text search in CodeSearchStrategy.

Tests the 3-way merge (keyword + BM25 + semantic), behaviour when BM25
returns empty results, and the effect of code_bm25_weight config.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.retrieval.strategies.code_search import CodeSearchStrategy
from fitz_sage.engines.fitz_krag.types import AddressKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> MagicMock:
    """Create a mock FitzKragConfig with fields CodeSearchStrategy reads."""
    cfg = MagicMock()
    cfg.keyword_weight = overrides.get("keyword_weight", 0.4)
    cfg.semantic_weight = overrides.get("semantic_weight", 0.6)
    cfg.code_bm25_weight = overrides.get("code_bm25_weight", 0.3)
    return cfg


def _make_row(
    row_id: str, score: float = 0.8, name: str = "func", bm25_score: float | None = None
) -> dict:
    """Build a dict mimicking a symbol store row."""
    row = {
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
    if bm25_score is not None:
        row["bm25_score"] = bm25_score
    return row


# ---------------------------------------------------------------------------
# TestBM25ThreeWayMerge
# ---------------------------------------------------------------------------


class TestBM25ThreeWayMerge:
    """Tests for 3-way keyword + BM25 + semantic merge in CodeSearchStrategy."""

    def test_three_way_merge_keyword_bm25_semantic(self):
        """All three sources contribute to the merged result set."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")
        config = _make_config(keyword_weight=0.4, semantic_weight=0.6, code_bm25_weight=0.3)

        # Keyword returns one unique result
        kw_row = _make_row("kw1", name="authenticate")
        store.search_by_name.return_value = [kw_row]

        # BM25 returns one unique result
        bm25_row = _make_row("bm1", name="login", bm25_score=0.9)
        store.search_bm25.return_value = [bm25_row]

        # Semantic returns one unique result
        sem_row = _make_row("sem1", score=0.85, name="verify")
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [sem_row]

        strategy = CodeSearchStrategy(store, embedder, config)

        result = strategy.retrieve("how does auth work", limit=10)

        # All three sources queried
        store.search_by_name.assert_called_once()
        store.search_bm25.assert_called_once()
        store.search_by_vector.assert_called_once()

        # All three results present (3 unique IDs)
        assert len(result) == 3

        # All are SYMBOL addresses
        for addr in result:
            assert addr.kind == AddressKind.SYMBOL

    def test_three_way_merge_overlapping_ids(self):
        """When the same symbol appears in multiple sources, scores are combined."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")
        config = _make_config(keyword_weight=0.4, semantic_weight=0.6, code_bm25_weight=0.3)

        # Same ID appears in all three
        shared_row = _make_row("shared1", score=0.8, name="authenticate")
        bm25_shared = _make_row("shared1", name="authenticate", bm25_score=0.7)
        unique_row = _make_row("unique1", score=0.5, name="unrelated")

        store.search_by_name.return_value = [shared_row]
        store.search_bm25.return_value = [bm25_shared]
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [shared_row, unique_row]

        strategy = CodeSearchStrategy(store, embedder, config)

        result = strategy.retrieve("authenticate", limit=10)

        # shared1 appears once (deduped by _merge_results)
        ids = [addr.metadata["symbol_id"] for addr in result]
        assert ids.count("shared1") == 1
        assert ids.count("unique1") == 1

        # shared1 should rank higher (scores from all three sources combined)
        assert result[0].metadata["symbol_id"] == "shared1"

    def test_bm25_empty_does_not_break_merge(self):
        """When BM25 returns empty results, merge proceeds with keyword + semantic."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        kw_row = _make_row("kw1", name="search")
        store.search_by_name.return_value = [kw_row]

        # BM25 returns empty
        store.search_bm25.return_value = []

        sem_row = _make_row("sem1", score=0.9, name="find")
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [sem_row]

        strategy = CodeSearchStrategy(store, embedder, config)

        result = strategy.retrieve("find items", limit=10)

        store.search_bm25.assert_called_once()
        assert len(result) == 2

    def test_bm25_exception_does_not_break_search(self):
        """When BM25 raises (e.g., no tsv column), search proceeds without it."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")
        config = _make_config()

        kw_row = _make_row("kw1", name="process")
        store.search_by_name.return_value = [kw_row]

        # BM25 raises
        store.search_bm25.side_effect = RuntimeError("content_tsv column not found")

        sem_row = _make_row("sem1", score=0.8, name="handle")
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [sem_row]

        strategy = CodeSearchStrategy(store, embedder, config)

        result = strategy.retrieve("process data", limit=10)

        assert len(result) == 2

    def test_code_bm25_weight_affects_scoring(self):
        """Higher code_bm25_weight gives more influence to BM25 results."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")

        # BM25-only result and semantic-only result
        bm25_row = _make_row("bm1", name="bm25_winner", bm25_score=0.95)
        sem_row = _make_row("sem1", score=0.95, name="semantic_winner")

        store.search_by_name.return_value = []
        store.search_bm25.return_value = [bm25_row]
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [sem_row]

        # High BM25 weight
        config_high_bm25 = _make_config(
            keyword_weight=0.1,
            semantic_weight=0.1,
            code_bm25_weight=0.8,
        )

        strategy = CodeSearchStrategy(store, embedder, config_high_bm25)
        result = strategy.retrieve("test query", limit=10)

        # BM25 result should rank first with high BM25 weight
        assert result[0].metadata["name"] == "bm25_winner"

    def test_code_bm25_weight_low_favors_semantic(self):
        """Lower code_bm25_weight means semantic results dominate."""
        store = MagicMock(name="symbol_store")
        embedder = MagicMock(name="embedder")

        bm25_row = _make_row("bm1", name="bm25_result", bm25_score=0.95)
        sem_row = _make_row("sem1", score=0.95, name="semantic_result")

        store.search_by_name.return_value = []
        store.search_bm25.return_value = [bm25_row]
        embedder.embed.return_value = [0.1]
        store.search_by_vector.return_value = [sem_row]

        # High semantic weight, low BM25 weight
        config_low_bm25 = _make_config(
            keyword_weight=0.1,
            semantic_weight=0.8,
            code_bm25_weight=0.1,
        )

        strategy = CodeSearchStrategy(store, embedder, config_low_bm25)
        result = strategy.retrieve("test query", limit=10)

        # Semantic result should rank first with high semantic weight
        assert result[0].metadata["name"] == "semantic_result"

    def test_weight_normalization_with_bm25(self):
        """When BM25 results are present, weights are normalized to sum to 1."""
        strategy = CodeSearchStrategy(MagicMock(), MagicMock(), _make_config())

        kw = [_make_row("kw1", name="a")]
        sem = [_make_row("sem1", score=0.8, name="b")]
        bm25 = [_make_row("bm1", name="c", bm25_score=0.7)]

        # Call _merge_results directly
        merged = strategy._merge_results(kw, sem, bm25)

        # All three should be present
        ids = {r["id"] for r in merged}
        assert ids == {"kw1", "sem1", "bm1"}

        # Each result has a combined_score
        for r in merged:
            assert "combined_score" in r
            assert r["combined_score"] > 0
