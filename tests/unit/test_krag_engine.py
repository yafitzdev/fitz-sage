# tests/unit/test_krag_engine.py
"""
Unit tests for FitzKragEngine.

All dependencies are mocked. The engine is constructed via __new__ with
mocked internals for answer/ingest/config tests. Only the init tests
exercise the real __init__ with patched imports.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fitz_ai.core import (
    Answer,
    ConfigurationError,
    GenerationError,
    KnowledgeError,
    Provenance,
    QueryError,
)
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
    engine._assembler = MagicMock(name="assembler")
    engine._synthesizer = MagicMock(name="synthesizer")
    return engine


def _make_query(text: str = "How does auth work?") -> MagicMock:
    """Return a mock Query with the given text."""
    q = MagicMock(name="query")
    q.text = text
    return q


# ---------------------------------------------------------------------------
# TestEngineInit
# ---------------------------------------------------------------------------


class TestEngineInit:
    """Tests that exercise the real __init__ / _init_components path."""

    # All lazy imports inside _init_components must be patched at the
    # location where they are imported (the engine module's namespace).
    _ENGINE_MOD = "fitz_ai.engines.fitz_krag.engine"

    @pytest.fixture()
    def _patches(self):
        """
        Context-manager that patches every lazy import used by
        _init_components and yields a dict of the mock objects.
        """
        names = {
            # llm
            "get_chat": f"fitz_ai.llm.client.get_chat",
            "get_embedder": f"fitz_ai.llm.client.get_embedder",
            # storage
            "PostgresConnectionManager": ("fitz_ai.storage.postgres.PostgresConnectionManager"),
            # stores
            "RawFileStore": ("fitz_ai.engines.fitz_krag.ingestion" ".raw_file_store.RawFileStore"),
            "SymbolStore": ("fitz_ai.engines.fitz_krag.ingestion" ".symbol_store.SymbolStore"),
            "ImportGraphStore": (
                "fitz_ai.engines.fitz_krag.ingestion" ".import_graph_store.ImportGraphStore"
            ),
            "SectionStore": ("fitz_ai.engines.fitz_krag.ingestion" ".section_store.SectionStore"),
            "ensure_schema": ("fitz_ai.engines.fitz_krag.ingestion" ".schema.ensure_schema"),
            # strategies
            "CodeSearchStrategy": (
                "fitz_ai.engines.fitz_krag.retrieval" ".strategies.code_search.CodeSearchStrategy"
            ),
            "SectionSearchStrategy": (
                "fitz_ai.engines.fitz_krag.retrieval"
                ".strategies.section_search.SectionSearchStrategy"
            ),
            # retrieval
            "RetrievalRouter": ("fitz_ai.engines.fitz_krag.retrieval" ".router.RetrievalRouter"),
            "ContentReader": ("fitz_ai.engines.fitz_krag.retrieval" ".reader.ContentReader"),
            "CodeExpander": ("fitz_ai.engines.fitz_krag.retrieval" ".expander.CodeExpander"),
            # query analysis
            "QueryAnalyzer": ("fitz_ai.engines.fitz_krag" ".query_analyzer.QueryAnalyzer"),
            # context + generation
            "ContextAssembler": ("fitz_ai.engines.fitz_krag.context" ".assembler.ContextAssembler"),
            "CodeSynthesizer": (
                "fitz_ai.engines.fitz_krag.generation" ".synthesizer.CodeSynthesizer"
            ),
        }

        patchers = {key: patch(target) for key, target in names.items()}
        mocks = {}
        for key, p in patchers.items():
            mocks[key] = p.start()

        # get_embedder returns a mock with a `.dimensions` attribute
        mocks["get_embedder"].return_value.dimensions = 1024
        # PostgresConnectionManager.get_instance() returns a mock
        mocks["PostgresConnectionManager"].get_instance.return_value = MagicMock(name="pg_instance")

        yield mocks

        for p in patchers.values():
            p.stop()

    def test_init_creates_components(self, _patches):
        """Engine initialises without error when all deps are available."""
        config = _make_config()
        engine = FitzKragEngine(config)

        # Core clients called
        _patches["get_chat"].assert_called_once()
        _patches["get_embedder"].assert_called_once()
        _patches["PostgresConnectionManager"].get_instance.assert_called_once()

        # Schema ensured
        _patches["ensure_schema"].assert_called_once()

        # Strategies and router created
        _patches["CodeSearchStrategy"].assert_called_once()
        _patches["SectionSearchStrategy"].assert_called_once()
        _patches["RetrievalRouter"].assert_called_once()

        # Reader, expander, analyzer, assembler, synthesizer created
        _patches["ContentReader"].assert_called_once()
        _patches["CodeExpander"].assert_called_once()
        _patches["QueryAnalyzer"].assert_called_once()
        _patches["ContextAssembler"].assert_called_once()
        _patches["CodeSynthesizer"].assert_called_once()

        # Config stored correctly
        assert engine.config is config

    def test_init_failure_raises_configuration_error(self):
        """
        If _init_components raises, __init__ wraps it as
        ConfigurationError.
        """
        with patch.object(
            FitzKragEngine,
            "_init_components",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(ConfigurationError, match="boom"):
                FitzKragEngine(_make_config())


# ---------------------------------------------------------------------------
# TestAnswer
# ---------------------------------------------------------------------------


class TestAnswer:
    """Tests for the answer() pipeline."""

    def test_answer_full_flow(self):
        """Happy path: every stage returns valid data."""
        engine = _make_engine()
        query = _make_query("What does the login function do?")

        # Wire up the pipeline stages
        analysis = MagicMock(name="analysis")
        engine._query_analyzer.analyze.return_value = analysis

        address_1 = MagicMock(name="addr1")
        address_2 = MagicMock(name="addr2")
        engine._retrieval_router.retrieve.return_value = [
            address_1,
            address_2,
        ]

        read_1 = MagicMock(name="read1")
        engine._reader.read.return_value = [read_1]

        expanded = [MagicMock(name="expanded1")]
        engine._expander.expand.return_value = expanded

        context = MagicMock(name="context")
        engine._assembler.assemble.return_value = context

        expected_answer = Answer(
            text="The login function authenticates users.",
            provenance=[Provenance(source_id="auth.py:42")],
            metadata={"engine": "fitz_krag"},
        )
        engine._synthesizer.generate.return_value = expected_answer

        # Execute
        result = engine.answer(query)

        # Verify each stage called with correct args
        engine._query_analyzer.analyze.assert_called_once_with(
            query.text,
        )
        engine._retrieval_router.retrieve.assert_called_once_with(
            query.text,
            analysis,
        )
        engine._reader.read.assert_called_once_with(
            [address_1, address_2],
            engine._config.top_read,
        )
        engine._expander.expand.assert_called_once_with([read_1])
        engine._assembler.assemble.assert_called_once_with(
            query.text,
            expanded,
        )
        engine._synthesizer.generate.assert_called_once_with(
            query.text,
            context,
            expanded,
        )

        assert result is expected_answer

    def test_answer_empty_query_raises(self):
        """Empty or whitespace-only query text raises QueryError."""
        engine = _make_engine()

        for blank in ("", "   ", "\t\n"):
            q = _make_query(blank)
            with pytest.raises(QueryError, match="empty"):
                engine.answer(q)

    def test_answer_no_addresses_returns_fallback(self):
        """Router returning [] yields a no-results fallback Answer."""
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.return_value = []

        result = engine.answer(query)

        assert "No relevant code" in result.text
        assert result.provenance == []
        assert result.metadata["engine"] == "fitz_krag"
        assert result.metadata["query"] == query.text

        # Reader should never be called
        engine._reader.read.assert_not_called()

    def test_answer_no_read_results_returns_fallback(self):
        """Reader returning [] yields a symbols-found fallback Answer."""
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = []

        result = engine.answer(query)

        assert "Found matching symbols" in result.text
        assert result.provenance == []
        assert result.metadata["engine"] == "fitz_krag"

        # Expander should never be called
        engine._expander.expand.assert_not_called()

    def test_answer_retrieval_error_raises_knowledge_error(self):
        """
        When the retrieval router raises an error whose message
        contains 'search', it is wrapped as KnowledgeError.
        """
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.side_effect = RuntimeError(
            "vector search connection timeout"
        )

        with pytest.raises(KnowledgeError, match="Retrieval failed"):
            engine.answer(query)

    def test_answer_retrieval_error_with_retriev_keyword(self):
        """
        Error message containing 'retriev' also maps to
        KnowledgeError.
        """
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.side_effect = RuntimeError("retrieval timeout")

        with pytest.raises(KnowledgeError, match="Retrieval failed"):
            engine.answer(query)

    def test_answer_generation_error_raises(self):
        """
        When the synthesizer raises an error whose message contains
        'generation', it is wrapped as GenerationError.
        """
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.side_effect = RuntimeError(
            "generation failed: LLM returned empty"
        )

        with pytest.raises(GenerationError, match="Generation failed"):
            engine.answer(query)

    def test_answer_llm_error_raises_generation_error(self):
        """
        Error message containing 'llm' maps to GenerationError.
        """
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.side_effect = RuntimeError("llm api rate limit exceeded")

        with pytest.raises(GenerationError, match="Generation failed"):
            engine.answer(query)

    def test_answer_unknown_error_raises_knowledge_error(self):
        """
        Errors that don't match 'retriev', 'search', 'generat',
        or 'llm' are wrapped as KnowledgeError with 'KRAG pipeline
        error' message.
        """
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.side_effect = RuntimeError("unexpected null pointer")

        with pytest.raises(KnowledgeError, match="KRAG pipeline error"):
            engine.answer(query)


# ---------------------------------------------------------------------------
# TestIngest
# ---------------------------------------------------------------------------


class TestIngest:
    """Tests for the ingest() method."""

    @patch("fitz_ai.engines.fitz_krag.ingestion.pipeline.KragIngestPipeline")
    def test_ingest_delegates_to_pipeline(self, mock_pipeline_cls):
        """ingest() creates a KragIngestPipeline and calls .ingest()."""
        engine = _make_engine()
        source = Path("/repo/src")
        expected_stats = {"files": 5, "symbols": 42, "imports": 10}
        mock_pipeline_cls.return_value.ingest.return_value = expected_stats

        result = engine.ingest(source)

        mock_pipeline_cls.assert_called_once_with(
            config=engine._config,
            chat=engine._chat,
            embedder=engine._embedder,
            connection_manager=engine._connection_manager,
            collection=engine._config.collection,
        )
        mock_pipeline_cls.return_value.ingest.assert_called_once_with(
            source,
        )
        assert result == expected_stats

    @patch("fitz_ai.engines.fitz_krag.ingestion.pipeline.KragIngestPipeline")
    def test_ingest_uses_collection_override(self, mock_pipeline_cls):
        """
        When a collection override is provided, it is used instead of
        the config default.
        """
        engine = _make_engine(collection="default_col")
        source = Path("/repo/src")
        mock_pipeline_cls.return_value.ingest.return_value = {}

        engine.ingest(source, collection="override_col")

        call_kwargs = mock_pipeline_cls.call_args
        assert call_kwargs.kwargs["collection"] == "override_col"

    @patch("fitz_ai.engines.fitz_krag.ingestion.pipeline.KragIngestPipeline")
    def test_ingest_none_collection_uses_config(self, mock_pipeline_cls):
        """
        When collection is None, the config's collection is used.
        """
        engine = _make_engine(collection="from_config")
        source = Path("/repo/src")
        mock_pipeline_cls.return_value.ingest.return_value = {}

        engine.ingest(source, collection=None)

        call_kwargs = mock_pipeline_cls.call_args
        assert call_kwargs.kwargs["collection"] == "from_config"


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for the config property."""

    def test_config_property_returns_config(self):
        """The config property returns the stored FitzKragConfig."""
        engine = _make_engine(collection="my_project")
        assert isinstance(engine.config, FitzKragConfig)
        assert engine.config.collection == "my_project"

    def test_config_property_reflects_overrides(self):
        """Config overrides are reflected in the property."""
        engine = _make_engine(
            collection="custom",
            top_read=10,
            top_addresses=20,
        )
        assert engine.config.top_read == 10
        assert engine.config.top_addresses == 20
        assert engine.config.collection == "custom"
