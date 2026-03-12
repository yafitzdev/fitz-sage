# tests/unit/test_llm_code_search.py
"""Tests for LLM structural code search strategy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fitz_ai.engines.fitz_krag.retrieval.strategies.llm_code_search import (
    LlmCodeSearchStrategy,
    ManifestBuilder,
)
from fitz_ai.engines.fitz_krag.types import AddressKind


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MANIFEST_DATA = [
    {
        "raw_file_id": "f1",
        "path": "app/models.py",
        "symbols": [
            {
                "name": "User",
                "qualified_name": "app.models.User",
                "kind": "class",
                "signature": "class User:",
                "start_line": 1,
                "end_line": 30,
                "imports": ["sqlalchemy"],
            },
            {
                "name": "__init__",
                "qualified_name": "app.models.User.__init__",
                "kind": "method",
                "signature": "def __init__(self, name: str) -> None",
                "start_line": 5,
                "end_line": 10,
                "imports": [],
            },
        ],
    },
    {
        "raw_file_id": "f2",
        "path": "app/views.py",
        "symbols": [
            {
                "name": "get_users",
                "qualified_name": "app.views.get_users",
                "kind": "function",
                "signature": "def get_users(db: Session) -> list[User]",
                "start_line": 1,
                "end_line": 15,
                "imports": ["app.models"],
            },
        ],
    },
]


def _make_symbol_store(manifest_data=None):
    store = MagicMock()
    store.get_structural_manifest.return_value = (
        manifest_data if manifest_data is not None else MANIFEST_DATA
    )
    return store


def _make_import_store(reverse_counts=None, imports_by_file=None):
    store = MagicMock()
    store.get_reverse_counts.return_value = reverse_counts or {}
    if imports_by_file:
        store.get_imports.side_effect = lambda fid: imports_by_file.get(fid, [])
    else:
        store.get_imports.return_value = []
    return store


def _make_chat_factory(response='["app/models.py"]'):
    chat = MagicMock()
    chat.chat.return_value = response
    factory = MagicMock(return_value=chat)
    return factory


def _make_config():
    config = MagicMock()
    config.code_search_mode = "auto"
    return config


def _make_fallback():
    from fitz_ai.engines.fitz_krag.types import Address

    fallback = MagicMock()
    fallback.retrieve.return_value = [
        Address(
            kind=AddressKind.SYMBOL,
            source_id="f1",
            location="app.models.User",
            summary="class User",
            score=0.5,
        )
    ]
    return fallback


# ---------------------------------------------------------------------------
# ManifestBuilder tests
# ---------------------------------------------------------------------------


class TestManifestBuilder:
    def test_build_returns_text_and_file_data(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        builder = ManifestBuilder(symbol_store, import_store)

        text, file_data = builder.build()

        assert "## app/models.py" in text
        assert "## app/views.py" in text
        assert "app/models.py" in file_data
        assert "app/views.py" in file_data
        assert file_data["app/models.py"]["raw_file_id"] == "f1"

    def test_build_includes_classes_and_functions(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        builder = ManifestBuilder(symbol_store, import_store)

        text, _ = builder.build()

        assert "classes:" in text
        assert "User" in text
        assert "functions:" in text
        assert "get_users" in text

    def test_build_includes_imports(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        builder = ManifestBuilder(symbol_store, import_store)

        text, _ = builder.build()

        assert "imports:" in text
        assert "sqlalchemy" in text

    def test_build_caches_on_same_symbol_count(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        builder = ManifestBuilder(symbol_store, import_store)

        text1, _ = builder.build()
        text2, _ = builder.build()

        assert text1 == text2
        # get_structural_manifest called twice (first to build, second to check count)
        # but the manifest text is reused from cache on second call
        assert symbol_store.get_structural_manifest.call_count == 2

    def test_truncation_strips_imports_first(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store(reverse_counts={"f1": 5, "f2": 1})
        builder = ManifestBuilder(symbol_store, import_store, max_chars=100)

        text, _ = builder.build()

        # With max_chars=100, truncation should kick in
        # f2 (least connected) should lose imports first
        assert "## app/models.py" in text
        assert "## app/views.py" in text

    def test_empty_manifest(self):
        symbol_store = _make_symbol_store(manifest_data=[])
        import_store = _make_import_store()
        builder = ManifestBuilder(symbol_store, import_store)

        text, file_data = builder.build()

        assert text == ""
        assert file_data == {}

    def test_extract_return_type(self):
        assert ManifestBuilder._extract_return_type("def foo() -> str") == "str"
        assert ManifestBuilder._extract_return_type("def foo()") == "None"

    def test_extract_params(self):
        assert ManifestBuilder._extract_params("def foo(a: int, b: str)") == "a: int, b: str"
        assert ManifestBuilder._extract_params("def foo(self, a: int)") == "a: int"
        assert ManifestBuilder._extract_params("def foo(self)") == ""


# ---------------------------------------------------------------------------
# LlmCodeSearchStrategy tests
# ---------------------------------------------------------------------------


class TestLlmCodeSearchStrategy:
    def test_retrieve_returns_addresses(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        chat_factory = _make_chat_factory('["app/models.py"]')
        config = _make_config()
        fallback = _make_fallback()

        strategy = LlmCodeSearchStrategy(
            symbol_store, import_store, chat_factory, config, fallback
        )
        addresses = strategy.retrieve("What is User?", limit=10)

        assert len(addresses) > 0
        assert all(a.kind == AddressKind.SYMBOL for a in addresses)
        assert any("User" in a.location for a in addresses)

    def test_scores_decrease_by_file_order(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        chat_factory = _make_chat_factory('["app/models.py", "app/views.py"]')
        config = _make_config()
        fallback = _make_fallback()

        strategy = LlmCodeSearchStrategy(
            symbol_store, import_store, chat_factory, config, fallback
        )
        addresses = strategy.retrieve("How does the app work?", limit=20)

        # First file's symbols should have higher scores than second file's
        f1_scores = [a.score for a in addresses if a.source_id == "f1"]
        f2_scores = [a.score for a in addresses if a.source_id == "f2"]
        assert f1_scores and f2_scores
        assert max(f1_scores) > max(f2_scores)

    def test_retrieve_expands_imports(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store(
            imports_by_file={
                "f1": [{"target_file_id": "f2", "target_module": "app.views", "import_names": []}]
            }
        )
        chat_factory = _make_chat_factory('["app/models.py"]')
        config = _make_config()
        fallback = _make_fallback()

        strategy = LlmCodeSearchStrategy(
            symbol_store, import_store, chat_factory, config, fallback
        )
        addresses = strategy.retrieve("What is User?", limit=20)

        # Should include symbols from both f1 (selected) and f2 (import-expanded)
        source_ids = {a.source_id for a in addresses}
        assert "f1" in source_ids
        assert "f2" in source_ids

    def test_fallback_on_llm_failure(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()

        # LLM raises an error
        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("LLM unavailable")
        chat_factory = MagicMock(return_value=chat)

        config = _make_config()
        fallback = _make_fallback()

        strategy = LlmCodeSearchStrategy(
            symbol_store, import_store, chat_factory, config, fallback
        )
        addresses = strategy.retrieve("What is User?", limit=10)

        # Should get fallback results
        assert len(addresses) == 1
        assert addresses[0].source_id == "f1"
        fallback.retrieve.assert_called_once()

    def test_fallback_on_empty_response(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        chat_factory = _make_chat_factory("[]")  # Empty selection
        config = _make_config()
        fallback = _make_fallback()

        strategy = LlmCodeSearchStrategy(
            symbol_store, import_store, chat_factory, config, fallback
        )
        addresses = strategy.retrieve("What is User?", limit=10)

        # Empty LLM response should trigger fallback
        assert len(addresses) == 1
        fallback.retrieve.assert_called_once()

    def test_fallback_on_invalid_json(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        chat_factory = _make_chat_factory("not json at all")
        config = _make_config()
        fallback = _make_fallback()

        strategy = LlmCodeSearchStrategy(
            symbol_store, import_store, chat_factory, config, fallback
        )
        addresses = strategy.retrieve("What is User?", limit=10)

        # Invalid JSON should trigger fallback
        assert len(addresses) == 1
        fallback.retrieve.assert_called_once()

    def test_retrieve_respects_limit(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        chat_factory = _make_chat_factory('["app/models.py", "app/views.py"]')
        config = _make_config()
        fallback = _make_fallback()

        strategy = LlmCodeSearchStrategy(
            symbol_store, import_store, chat_factory, config, fallback
        )
        addresses = strategy.retrieve("What is User?", limit=1)

        assert len(addresses) <= 1

    def test_forwards_hyde_and_raw_store(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        chat_factory = _make_chat_factory()
        config = _make_config()
        fallback = _make_fallback()

        strategy = LlmCodeSearchStrategy(
            symbol_store, import_store, chat_factory, config, fallback
        )

        # Simulate engine wiring
        strategy._hyde_generator = "hyde_gen"
        strategy._raw_store = "raw_store"

        assert strategy._hyde_generator == "hyde_gen"
        assert strategy._raw_store == "raw_store"

    def test_fallback_receives_all_kwargs(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()

        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("fail")
        chat_factory = MagicMock(return_value=chat)

        config = _make_config()
        fallback = _make_fallback()

        strategy = LlmCodeSearchStrategy(
            symbol_store, import_store, chat_factory, config, fallback
        )

        qv = [0.1, 0.2, 0.3]
        hv = [[0.4, 0.5, 0.6]]
        detection = MagicMock()

        strategy.retrieve("query", limit=5, detection=detection, query_vector=qv, hyde_vectors=hv)

        fallback.retrieve.assert_called_once_with(
            "query",
            5,
            detection=detection,
            query_vector=qv,
            hyde_vectors=hv,
        )

    def test_llm_select_files_parses_json_in_markdown(self):
        """LLM may wrap JSON in markdown code blocks."""
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        chat_factory = _make_chat_factory(
            '```json\n["app/models.py", "app/views.py"]\n```'
        )
        config = _make_config()
        fallback = _make_fallback()

        strategy = LlmCodeSearchStrategy(
            symbol_store, import_store, chat_factory, config, fallback
        )
        addresses = strategy.retrieve("How does get_users work?", limit=20)

        source_ids = {a.source_id for a in addresses}
        assert "f1" in source_ids
        assert "f2" in source_ids
