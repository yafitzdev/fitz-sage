# tests/unit/test_krag_query_rewriting.py
"""
Unit tests for query rewriting in FitzKragEngine.

Tests that the engine's _query_rewriter (when present) is called during the
answer() pipeline, and that failures or absence are handled gracefully.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_ai.core import Answer, Provenance
from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_ai.engines.fitz_krag.engine import FitzKragEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> FitzKragConfig:
    """Create a minimal FitzKragConfig for testing."""
    defaults = {"collection": "test_collection"}
    defaults.update(overrides)
    return FitzKragConfig(**defaults)


def _make_engine(**config_overrides) -> FitzKragEngine:
    """
    Build a FitzKragEngine with every component replaced by a MagicMock.

    Bypasses __init__ entirely so no real imports are triggered.
    """
    config = _make_config(**config_overrides)
    engine = FitzKragEngine.__new__(FitzKragEngine)
    engine._config = config
    engine._chat = MagicMock(name="chat")
    engine._embedder = MagicMock(name="embedder")
    engine._connection_manager = MagicMock(name="connection_manager")
    engine._raw_store = MagicMock(name="raw_store")
    engine._symbol_store = MagicMock(name="symbol_store")
    engine._import_store = MagicMock(name="import_store")
    engine._section_store = MagicMock(name="section_store")
    engine._query_analyzer = MagicMock(name="query_analyzer")
    engine._retrieval_router = MagicMock(name="retrieval_router")
    engine._reader = MagicMock(name="reader")
    engine._expander = MagicMock(name="expander")
    engine._table_handler = MagicMock(name="table_handler")
    engine._table_handler.process.side_effect = lambda q, results: results
    engine._assembler = MagicMock(name="assembler")
    engine._synthesizer = MagicMock(name="synthesizer")
    engine._constraints = []
    engine._governor = None
    engine._cloud_client = None
    engine._detection_orchestrator = None
    engine._query_rewriter = None
    engine._address_reranker = None
    engine._hop_controller = None
    engine._table_store = MagicMock(name="table_store")
    engine._pg_table_store = MagicMock(name="pg_table_store")
    engine._chat_factory = None
    engine._vocabulary_store = None
    engine._keyword_matcher = None
    engine._entity_graph_store = None
    engine._bg_worker = None
    engine._manifest = None
    engine._source_dir = None
    engine._hyde_generator = None
    return engine


def _make_query(text: str = "How does auth work?") -> MagicMock:
    """Return a mock Query with the given text."""
    q = MagicMock(name="query")
    q.text = text
    return q


def _wire_happy_path(engine: FitzKragEngine, query_text: str) -> Answer:
    """Wire up all pipeline stages to return valid data for a full flow."""
    analysis = MagicMock(name="analysis")
    engine._query_analyzer.analyze.return_value = analysis

    address = MagicMock(name="addr")
    engine._retrieval_router.retrieve.return_value = [address]

    read_result = MagicMock(name="read")
    engine._reader.read.return_value = [read_result]
    engine._expander.expand.return_value = [read_result]

    context = MagicMock(name="context")
    engine._assembler.assemble.return_value = context

    expected = Answer(
        text="Answer text.",
        provenance=[Provenance(source_id="file.py:10")],
        metadata={"engine": "fitz_krag"},
    )
    engine._synthesizer.generate.return_value = expected
    return expected


# ---------------------------------------------------------------------------
# TestQueryRewriting
# ---------------------------------------------------------------------------


class TestQueryRewriting:
    """Tests for query rewriting integration in the answer() pipeline."""

    def test_rewrite_called_and_rewritten_query_used(self):
        """Rewrite runs in parallel with analysis; rewritten query used for retrieval."""
        engine = _make_engine()
        query = _make_query("How does the authentication system handle user login sessions securely?")

        # Set up rewriter
        rewriter = MagicMock(name="rewriter")
        rewrite_result = MagicMock()
        rewrite_result.rewritten_query = "authentication module implementation for secure user login session handling"
        rewriter.rewrite.return_value = rewrite_result
        engine._query_rewriter = rewriter

        expected = _wire_happy_path(engine, query.text)

        result = engine.answer(query)

        # Rewriter called with original query
        rewriter.rewrite.assert_called_once_with(query.text)

        # Analyzer receives original query (runs in parallel with rewrite)
        engine._query_analyzer.analyze.assert_called_once_with(query.text)

        # Router receives the rewritten query for retrieval
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        assert call_args[0] == (rewrite_result.rewritten_query, engine._query_analyzer.analyze.return_value)
        assert call_args[1]["rewrite_result"] is rewrite_result

        assert result is expected

    def test_original_query_used_when_rewrite_returns_same_text(self):
        """When rewriter returns the same text, original query flows through unchanged."""
        engine = _make_engine()
        query = _make_query("What is the login function and how does it validate user credentials?")

        rewriter = MagicMock(name="rewriter")
        rewrite_result = MagicMock()
        rewrite_result.rewritten_query = query.text
        rewriter.rewrite.return_value = rewrite_result
        engine._query_rewriter = rewriter

        expected = _wire_happy_path(engine, query.text)

        result = engine.answer(query)

        rewriter.rewrite.assert_called_once_with(query.text)

        # Analyzer and router use the original query (same as rewritten)
        engine._query_analyzer.analyze.assert_called_once_with(query.text)
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        assert call_args[0] == (query.text, engine._query_analyzer.analyze.return_value)
        assert call_args[1]["rewrite_result"] is rewrite_result

        assert result is expected

    def test_fallback_to_original_on_rewrite_error(self):
        """When rewriter raises an exception, the original query is used."""
        engine = _make_engine()
        query = _make_query("How does the authentication system work when handling multiple sessions?")

        rewriter = MagicMock(name="rewriter")
        rewriter.rewrite.side_effect = RuntimeError("LLM timeout")
        engine._query_rewriter = rewriter

        expected = _wire_happy_path(engine, query.text)

        result = engine.answer(query)

        rewriter.rewrite.assert_called_once_with(query.text)

        # Falls back to original query text
        engine._query_analyzer.analyze.assert_called_once_with(query.text)
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        assert call_args[0] == (query.text, engine._query_analyzer.analyze.return_value)
        assert call_args[1]["rewrite_result"] is None

        assert result is expected

    def test_rewriting_skipped_when_rewriter_is_none(self):
        """When _query_rewriter is None, the original query flows through directly."""
        engine = _make_engine()
        query = _make_query("Where is the UserService class defined and what methods does it expose?")
        assert engine._query_rewriter is None

        expected = _wire_happy_path(engine, query.text)

        result = engine.answer(query)

        # Analyzer uses original query directly
        engine._query_analyzer.analyze.assert_called_once_with(query.text)
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        assert call_args[0] == (query.text, engine._query_analyzer.analyze.return_value)
        assert call_args[1]["rewrite_result"] is None

        assert result is expected
