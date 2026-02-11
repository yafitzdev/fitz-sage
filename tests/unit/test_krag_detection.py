# tests/unit/test_krag_detection.py
"""
Unit tests for shared detection integration in the Fitz KRAG engine.

Covers:
- Detection orchestrator initialization (enabled/disabled)
- Router.retrieve behavior with and without detection
- Detection summary flowing through engine.answer() to the router
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_ai.engines.fitz_krag.engine import FitzKragEngine
from fitz_ai.engines.fitz_krag.retrieval.router import RetrievalRouter
from fitz_ai.engines.fitz_krag.types import Address, AddressKind

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
    Sets _detection_orchestrator, _cloud_client, _constraints, and _governor
    to None by default (detection/cloud/guardrails disabled).
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
    engine._detection_orchestrator = None
    engine._cloud_client = None
    engine._constraints = []
    engine._governor = None
    return engine


def _make_query(text: str = "How does auth work?") -> MagicMock:
    """Return a mock Query with the given text."""
    q = MagicMock(name="query")
    q.text = text
    return q


def _make_address(
    source_id: str = "src1",
    location: str = "mod.func",
    score: float = 0.9,
) -> Address:
    return Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location=location,
        summary="test",
        score=score,
    )


def _make_detection_summary(
    query_variations: list[str] | None = None,
    comparison_entities: list[str] | None = None,
    fetch_multiplier: int = 1,
) -> MagicMock:
    """Create a mock DetectionSummary with specified attributes."""
    detection = MagicMock(name="detection_summary")
    detection.query_variations = query_variations or []
    detection.comparison_entities = comparison_entities or []
    detection.fetch_multiplier = fetch_multiplier
    return detection


# ---------------------------------------------------------------------------
# TestEngineDetectionInit
# ---------------------------------------------------------------------------


class TestEngineDetectionInit:
    """Tests for detection orchestrator initialization."""

    def test_detection_disabled_has_none_orchestrator(self):
        """Engine with enable_detection=False has None detection_orchestrator."""
        engine = _make_engine(enable_detection=False)

        assert engine._detection_orchestrator is None

    @patch("fitz_ai.retrieval.detection.registry.DetectionOrchestrator")
    @patch("fitz_ai.llm.factory.get_chat_factory")
    @patch("fitz_ai.llm.client.get_chat")
    @patch("fitz_ai.llm.client.get_embedder")
    @patch("fitz_ai.storage.postgres.PostgresConnectionManager")
    @patch("fitz_ai.engines.fitz_krag.ingestion.raw_file_store.RawFileStore")
    @patch("fitz_ai.engines.fitz_krag.ingestion.symbol_store.SymbolStore")
    @patch("fitz_ai.engines.fitz_krag.ingestion.import_graph_store.ImportGraphStore")
    @patch("fitz_ai.engines.fitz_krag.ingestion.section_store.SectionStore")
    @patch("fitz_ai.engines.fitz_krag.ingestion.schema.ensure_schema")
    @patch("fitz_ai.engines.fitz_krag.retrieval.strategies.code_search.CodeSearchStrategy")
    @patch("fitz_ai.engines.fitz_krag.retrieval.strategies.section_search.SectionSearchStrategy")
    @patch("fitz_ai.engines.fitz_krag.retrieval.router.RetrievalRouter")
    @patch("fitz_ai.engines.fitz_krag.retrieval.reader.ContentReader")
    @patch("fitz_ai.engines.fitz_krag.retrieval.expander.CodeExpander")
    @patch("fitz_ai.engines.fitz_krag.query_analyzer.QueryAnalyzer")
    @patch("fitz_ai.engines.fitz_krag.context.assembler.ContextAssembler")
    @patch("fitz_ai.engines.fitz_krag.generation.synthesizer.CodeSynthesizer")
    def test_detection_enabled_creates_orchestrator(
        self,
        _synth,
        _asm,
        _qa,
        _exp,
        _reader,
        _router,
        _sect_strat,
        _code_strat,
        _ensure,
        _sect_store,
        _imp_store,
        _sym_store,
        _raw_store,
        mock_pg,
        mock_get_embedder,
        mock_get_chat,
        mock_get_chat_factory,
        mock_orchestrator_cls,
    ):
        """Engine with enable_detection=True creates a DetectionOrchestrator."""
        mock_get_embedder.return_value.dimensions = 1024
        mock_pg.get_instance.return_value = MagicMock(name="pg_instance")
        mock_factory = MagicMock(name="chat_factory")
        mock_get_chat_factory.return_value = mock_factory

        config = _make_config(enable_detection=True, enable_guardrails=False)
        engine = FitzKragEngine(config)

        mock_get_chat_factory.assert_called_once_with(config.chat)
        mock_orchestrator_cls.assert_called_once_with(chat_factory=mock_factory)
        assert engine._detection_orchestrator is mock_orchestrator_cls.return_value


# ---------------------------------------------------------------------------
# TestRouterDetection
# ---------------------------------------------------------------------------


class TestRouterDetection:
    """Tests for RetrievalRouter.retrieve with detection integration."""

    def _make_router(
        self,
        code_results: list[Address] | None = None,
        section_results: list[Address] | None = None,
        top_addresses: int = 10,
    ) -> RetrievalRouter:
        """Build a RetrievalRouter with mocked strategies."""
        code_strategy = MagicMock(name="code_strategy")
        code_strategy.retrieve.return_value = code_results or []

        section_strategy = MagicMock(name="section_strategy")
        section_strategy.retrieve.return_value = section_results or []

        config = _make_config(
            top_addresses=top_addresses,
            fallback_to_chunks=False,
        )

        return RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=config,
            section_strategy=section_strategy,
        )

    def test_retrieve_without_detection_backward_compat(self):
        """Router.retrieve without detection works as before (backward compat)."""
        addr = _make_address("file.py", "foo.bar", 0.8)
        router = self._make_router(code_results=[addr])

        results = router.retrieve("what is foo?")

        assert len(results) == 1
        assert results[0].source_id == "file.py"
        # Code strategy called once (only the original query)
        router._code_strategy.retrieve.assert_called_once()

    def test_retrieve_with_query_variations_runs_extra_queries(self):
        """Router.retrieve with query_variations runs additional queries."""
        addr1 = _make_address("a.py", "mod.alpha", 0.9)
        addr2 = _make_address("b.py", "mod.beta", 0.7)
        # First call returns addr1, second returns addr2
        code_strategy = MagicMock(name="code_strategy")
        code_strategy.retrieve.side_effect = [[addr1], [addr2]]

        section_strategy = MagicMock(name="section_strategy")
        section_strategy.retrieve.return_value = []

        config = _make_config(top_addresses=10, fallback_to_chunks=False)
        router = RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=config,
            section_strategy=section_strategy,
        )

        detection = _make_detection_summary(query_variations=["expanded query"])
        results = router.retrieve("original query", detection=detection)

        # Code strategy called twice: once for "original query", once for "expanded query"
        assert code_strategy.retrieve.call_count == 2
        calls = code_strategy.retrieve.call_args_list
        assert calls[0].args[0] == "original query"
        assert calls[1].args[0] == "expanded query"
        # Both addresses returned
        assert len(results) == 2

    def test_retrieve_with_comparison_entities_creates_augmented_queries(self):
        """Router.retrieve with comparison_entities creates entity-augmented queries."""
        addr1 = _make_address("a.py", "mod.alpha", 0.9)
        addr2 = _make_address("b.py", "mod.beta", 0.8)
        addr3 = _make_address("c.py", "mod.gamma", 0.7)

        code_strategy = MagicMock(name="code_strategy")
        code_strategy.retrieve.side_effect = [[addr1], [addr2], [addr3]]

        section_strategy = MagicMock(name="section_strategy")
        section_strategy.retrieve.return_value = []

        config = _make_config(top_addresses=10, fallback_to_chunks=False)
        router = RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=config,
            section_strategy=section_strategy,
        )

        detection = _make_detection_summary(
            comparison_entities=["React", "Vue"],
        )
        results = router.retrieve("compare frameworks", detection=detection)

        # Code strategy called 3 times: original + 2 entity-augmented queries
        assert code_strategy.retrieve.call_count == 3
        calls = code_strategy.retrieve.call_args_list
        assert calls[0].args[0] == "compare frameworks"
        assert calls[1].args[0] == "compare frameworks React"
        assert calls[2].args[0] == "compare frameworks Vue"

    def test_retrieve_with_fetch_multiplier_increases_limit(self):
        """Router.retrieve with fetch_multiplier increases the retrieval limit."""
        addr = _make_address("a.py", "mod.func", 0.9)

        code_strategy = MagicMock(name="code_strategy")
        code_strategy.retrieve.return_value = [addr]

        section_strategy = MagicMock(name="section_strategy")
        section_strategy.retrieve.return_value = []

        config = _make_config(top_addresses=5, fallback_to_chunks=False)
        router = RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=config,
            section_strategy=section_strategy,
        )

        detection = _make_detection_summary(fetch_multiplier=3)
        router.retrieve("query", detection=detection)

        # limit passed to code strategy should be 5 * 3 = 15
        call_args = code_strategy.retrieve.call_args
        assert call_args.args[1] == 15  # second positional arg is limit


# ---------------------------------------------------------------------------
# TestEngineAnswerDetectionFlow
# ---------------------------------------------------------------------------


class TestEngineAnswerDetectionFlow:
    """Tests for detection flowing through engine.answer() to the router."""

    def test_detection_result_flows_to_router(self):
        """Detection result from orchestrator is passed to router.retrieve."""
        engine = _make_engine(enable_detection=True)

        # Wire up detection orchestrator
        mock_orchestrator = MagicMock(name="detection_orchestrator")
        mock_detection = _make_detection_summary(query_variations=["expanded"])
        mock_orchestrator.detect_for_retrieval.return_value = mock_detection
        engine._detection_orchestrator = mock_orchestrator

        # Wire up pipeline stages
        query = _make_query("How does auth work?")
        analysis = MagicMock(name="analysis")
        engine._query_analyzer.analyze.return_value = analysis
        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.return_value = MagicMock(
            text="answer", provenance=[], metadata={}
        )

        engine.answer(query)

        # Detection orchestrator called with query text
        mock_orchestrator.detect_for_retrieval.assert_called_once_with(query.text)

        # Router called with detection result
        engine._retrieval_router.retrieve.assert_called_once_with(
            query.text, analysis, detection=mock_detection
        )

    def test_detection_disabled_stays_none_in_answer_flow(self):
        """When detection is disabled, detection stays None through answer flow."""
        engine = _make_engine(enable_detection=False)
        assert engine._detection_orchestrator is None

        # Wire up pipeline stages
        query = _make_query("How does auth work?")
        analysis = MagicMock(name="analysis")
        engine._query_analyzer.analyze.return_value = analysis
        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.return_value = MagicMock(
            text="answer", provenance=[], metadata={}
        )

        engine.answer(query)

        # Router called with detection=None (no detection)
        engine._retrieval_router.retrieve.assert_called_once_with(
            query.text, analysis, detection=None
        )
