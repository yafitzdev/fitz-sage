# tests/unit/test_krag_cloud.py
"""
Unit tests for FitzKragEngine cloud cache integration.

All dependencies are mocked. The engine is constructed via __new__ with
mocked internals. Tests cover cloud_client initialization, cache lookup
(hit/miss/error), cache store (success/error), and the answer() early-return
path on cache hit.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fitz_ai.cloud.cache_key import CacheVersions
from fitz_ai.cloud.client import CacheLookupResult
from fitz_ai.core import Answer, Provenance
from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_ai.engines.fitz_krag.engine import FitzKragEngine
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
    """
    config = _make_config(**config_overrides)
    engine = FitzKragEngine.__new__(FitzKragEngine)
    engine._config = config
    engine._chat = MagicMock(name="chat")
    engine._embedder = MagicMock(name="embedder")
    engine._embedder.embed.return_value = [0.1, 0.2, 0.3]
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
    engine._table_handler = MagicMock(name="table_handler")
    engine._table_handler.process.side_effect = lambda q, results: results
    engine._table_store = MagicMock(name="table_store")
    engine._pg_table_store = MagicMock(name="pg_table_store")
    engine._constraints = []
    engine._governor = None
    engine._cloud_client = None
    engine._detection_orchestrator = None
    engine._query_rewriter = None
    engine._address_reranker = None
    engine._hop_controller = None
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


def _make_address(source_id: str = "auth.py:login") -> Address:
    """Return a concrete Address for cache tests."""
    return Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location="auth.py",
        summary="Login function",
        score=0.95,
    )


def _make_answer(text: str = "The login function authenticates users.") -> Answer:
    """Return a concrete Answer for cache tests."""
    return Answer(
        text=text,
        provenance=[Provenance(source_id="auth.py:42")],
        metadata={"engine": "fitz_krag"},
    )


# ---------------------------------------------------------------------------
# TestCloudClientInit
# ---------------------------------------------------------------------------


class TestCloudClientInit:
    """Tests for cloud_client presence based on config."""

    def test_cloud_disabled_by_default(self):
        """Engine with default (empty) cloud config has None cloud_client."""
        engine = _make_engine()

        assert engine._cloud_client is None

    def test_cloud_enabled_creates_client(self):
        """
        Engine with a cloud dict containing enabled=True creates a
        CloudClient instance during _init_components.
        """
        from contextlib import ExitStack

        mock_cloud_client_cls = MagicMock(name="CloudClient")
        mock_cloud_config_cls = MagicMock(name="CloudConfig")
        mock_cloud_config_instance = mock_cloud_config_cls.return_value
        mock_cloud_config_instance.org_id = "org-123"

        patches = [
            patch("fitz_ai.llm.client.get_chat", return_value=MagicMock()),
            patch("fitz_ai.llm.client.get_embedder", return_value=MagicMock(dimensions=1024)),
            patch(
                "fitz_ai.storage.postgres.PostgresConnectionManager.get_instance",
                return_value=MagicMock(),
            ),
            patch("fitz_ai.engines.fitz_krag.ingestion.raw_file_store.RawFileStore"),
            patch("fitz_ai.engines.fitz_krag.ingestion.symbol_store.SymbolStore"),
            patch("fitz_ai.engines.fitz_krag.ingestion.import_graph_store.ImportGraphStore"),
            patch("fitz_ai.engines.fitz_krag.ingestion.section_store.SectionStore"),
            patch("fitz_ai.engines.fitz_krag.ingestion.table_store.TableStore"),
            patch("fitz_ai.engines.fitz_krag.ingestion.schema.ensure_schema"),
            patch("fitz_ai.engines.fitz_krag.retrieval.strategies.code_search.CodeSearchStrategy"),
            patch(
                "fitz_ai.engines.fitz_krag.retrieval.strategies.section_search.SectionSearchStrategy"
            ),
            patch(
                "fitz_ai.engines.fitz_krag.retrieval.strategies.table_search.TableSearchStrategy"
            ),
            patch("fitz_ai.engines.fitz_krag.retrieval.router.RetrievalRouter"),
            patch("fitz_ai.engines.fitz_krag.retrieval.reader.ContentReader"),
            patch("fitz_ai.engines.fitz_krag.retrieval.expander.CodeExpander"),
            patch("fitz_ai.engines.fitz_krag.retrieval.table_handler.TableQueryHandler"),
            patch("fitz_ai.engines.fitz_krag.query_analyzer.QueryAnalyzer"),
            patch("fitz_ai.engines.fitz_krag.context.assembler.ContextAssembler"),
            patch("fitz_ai.engines.fitz_krag.generation.synthesizer.CodeSynthesizer"),
            patch("fitz_ai.tabular.store.postgres.PostgresTableStore"),
            patch("fitz_ai.llm.factory.get_chat_factory"),
            patch("fitz_ai.cloud.client.CloudClient", mock_cloud_client_cls),
            patch("fitz_ai.cloud.config.CloudConfig", mock_cloud_config_cls),
        ]

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            cloud_dict = {"enabled": True, "api_key": "key", "org_key": "org"}
            config = _make_config(cloud=cloud_dict, enable_detection=False)
            engine = FitzKragEngine(config)

            mock_cloud_config_cls.assert_called_once_with(**cloud_dict)
            mock_cloud_config_instance.validate_config.assert_called_once()
            mock_cloud_client_cls.assert_called_once_with(
                mock_cloud_config_instance,
                mock_cloud_config_instance.org_id or "",
            )
            assert engine._cloud_client is mock_cloud_client_cls.return_value


# ---------------------------------------------------------------------------
# TestCheckCloudCache
# ---------------------------------------------------------------------------


class TestCheckCloudCache:
    """Tests for _check_cloud_cache method."""

    def test_cache_miss_returns_none(self):
        """_check_cloud_cache returns None on cache miss."""
        engine = _make_engine()
        engine._cloud_client = MagicMock(name="cloud_client")
        engine._cloud_client.lookup_cache.return_value = CacheLookupResult(hit=False)

        addresses = [_make_address("auth.py:login"), _make_address("auth.py:logout")]
        result = engine._check_cloud_cache("How does auth work?", addresses)

        assert result is None
        engine._cloud_client.lookup_cache.assert_called_once()

    def test_cache_hit_returns_answer(self):
        """_check_cloud_cache returns the cached Answer on hit."""
        engine = _make_engine()
        engine._cloud_client = MagicMock(name="cloud_client")
        cached_answer = _make_answer("Cached: login handles auth.")
        engine._cloud_client.lookup_cache.return_value = CacheLookupResult(
            hit=True, answer=cached_answer
        )

        addresses = [_make_address("auth.py:login")]
        result = engine._check_cloud_cache("How does auth work?", addresses)

        assert result is cached_answer
        assert result.text == "Cached: login handles auth."

    def test_cache_lookup_uses_embedder_and_fingerprint(self):
        """
        _check_cloud_cache embeds the query, computes retrieval fingerprint
        from address source_ids, and passes both to lookup_cache.
        """
        engine = _make_engine()
        engine._cloud_client = MagicMock(name="cloud_client")
        engine._cloud_client.lookup_cache.return_value = CacheLookupResult(hit=False)
        engine._embedder.embed.return_value = [0.5, 0.6, 0.7]

        addresses = [_make_address("a.py:foo"), _make_address("b.py:bar")]
        engine._check_cloud_cache("test query", addresses)

        # Embedder called with query text
        engine._embedder.embed.assert_called_once_with("test query", task_type="query")

        # lookup_cache called with embedding and fingerprint
        call_kwargs = engine._cloud_client.lookup_cache.call_args
        assert call_kwargs.kwargs["query_text"] == "test query"
        assert call_kwargs.kwargs["query_embedding"] == [0.5, 0.6, 0.7]
        assert isinstance(call_kwargs.kwargs["retrieval_fingerprint"], str)
        assert isinstance(call_kwargs.kwargs["versions"], CacheVersions)

    def test_cache_exception_returns_none_fail_open(self):
        """_check_cloud_cache returns None when an exception occurs (fail-open)."""
        engine = _make_engine()
        engine._cloud_client = MagicMock(name="cloud_client")
        engine._cloud_client.lookup_cache.side_effect = ConnectionError("Cloud API unreachable")

        addresses = [_make_address()]
        result = engine._check_cloud_cache("How does auth work?", addresses)

        assert result is None


# ---------------------------------------------------------------------------
# TestStoreCloudCache
# ---------------------------------------------------------------------------


class TestStoreCloudCache:
    """Tests for _store_cloud_cache method."""

    def test_store_calls_client_with_correct_args(self):
        """_store_cloud_cache calls store_cache with query, embedding, fingerprint, answer."""
        engine = _make_engine()
        engine._cloud_client = MagicMock(name="cloud_client")
        engine._embedder.embed.return_value = [0.1, 0.2, 0.3]

        addresses = [_make_address("auth.py:login")]
        answer = _make_answer()

        engine._store_cloud_cache("How does auth work?", addresses, answer)

        engine._cloud_client.store_cache.assert_called_once()
        call_kwargs = engine._cloud_client.store_cache.call_args.kwargs
        assert call_kwargs["query_text"] == "How does auth work?"
        assert call_kwargs["query_embedding"] == [0.1, 0.2, 0.3]
        assert isinstance(call_kwargs["retrieval_fingerprint"], str)
        assert isinstance(call_kwargs["versions"], CacheVersions)
        assert call_kwargs["answer"] is answer

    def test_store_uses_cached_embedding_when_available(self):
        """
        If _cached_query_embedding is set (from prior _check_cloud_cache),
        _store_cloud_cache reuses it instead of calling embedder again.
        """
        engine = _make_engine()
        engine._cloud_client = MagicMock(name="cloud_client")
        engine._cached_query_embedding = [0.9, 0.8, 0.7]

        addresses = [_make_address()]
        answer = _make_answer()

        engine._store_cloud_cache("test", addresses, answer)

        # Embedder should NOT be called since cached embedding exists
        engine._embedder.embed.assert_not_called()
        call_kwargs = engine._cloud_client.store_cache.call_args.kwargs
        assert call_kwargs["query_embedding"] == [0.9, 0.8, 0.7]

    def test_store_exception_handled_gracefully(self):
        """_store_cloud_cache swallows exceptions (fail-open, no raise)."""
        engine = _make_engine()
        engine._cloud_client = MagicMock(name="cloud_client")
        engine._cloud_client.store_cache.side_effect = ConnectionError("Cloud API unreachable")

        addresses = [_make_address()]
        answer = _make_answer()

        # Should not raise
        engine._store_cloud_cache("How does auth work?", addresses, answer)


# ---------------------------------------------------------------------------
# TestAnswerCloudCacheHit
# ---------------------------------------------------------------------------


class TestAnswerCloudCacheHit:
    """Tests for answer() early return on cloud cache hit."""

    def test_answer_returns_early_on_cache_hit(self):
        """
        When cloud_client is set and _check_cloud_cache returns an Answer,
        answer() returns that cached answer without calling reader/expander/
        assembler/synthesizer.
        """
        engine = _make_engine()
        engine._cloud_client = MagicMock(name="cloud_client")

        query = _make_query("How does auth work?")

        # Wire up pipeline stages before cloud check
        analysis = MagicMock(name="analysis")
        engine._query_analyzer.analyze.return_value = analysis

        address = _make_address()
        engine._retrieval_router.retrieve.return_value = [address]

        # Cloud cache returns a hit
        cached_answer = _make_answer("Cached answer from cloud")
        with patch.object(engine, "_check_cloud_cache", return_value=cached_answer):
            result = engine.answer(query)

        # The cached answer is returned directly
        assert result is cached_answer
        assert result.text == "Cached answer from cloud"

        # Pipeline stages after cloud check should NOT be called
        engine._reader.read.assert_not_called()
        engine._expander.expand.assert_not_called()
        engine._assembler.assemble.assert_not_called()
        engine._synthesizer.generate.assert_not_called()
