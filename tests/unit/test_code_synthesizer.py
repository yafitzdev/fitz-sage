# tests/unit/test_code_synthesizer.py
"""Tests for CodeSynthesizer: prompt building, provenance construction."""

from unittest.mock import MagicMock

import pytest

from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_ai.engines.fitz_krag.generation.synthesizer import (
    CodeSynthesizer,
)
from fitz_ai.engines.fitz_krag.types import Address, AddressKind, ReadResult


@pytest.fixture
def config():
    return FitzKragConfig(
        collection="test",
        enable_citations=True,
        strict_grounding=True,
    )


@pytest.fixture
def mock_chat():
    chat = MagicMock()
    chat.chat.return_value = "The function `hello()` returns 'world'. [S1]"
    return chat


def _make_result(kind="function", context_type=None):
    addr = Address(
        kind=AddressKind.SYMBOL,
        source_id="f1",
        location="mod.hello",
        summary="Say hello",
        metadata={"kind": kind, "start_line": 3, "end_line": 5},
    )
    meta = {"context_type": context_type} if context_type else {}
    return ReadResult(
        address=addr,
        content="def hello():\n    return 'world'",
        file_path="src/mod.py",
        line_range=(3, 5),
        metadata=meta,
    )


class TestCodeSynthesizer:
    def test_generate_returns_answer(self, mock_chat, config):
        synth = CodeSynthesizer(mock_chat, config)
        results = [_make_result()]
        answer = synth.generate("What does hello do?", "context...", results)

        assert answer.text == "The function `hello()` returns 'world'. [S1]"
        assert answer.metadata["engine"] == "fitz_krag"

    def test_provenance_built_correctly(self, mock_chat, config):
        synth = CodeSynthesizer(mock_chat, config)
        results = [_make_result()]
        answer = synth.generate("test", "context", results)

        assert len(answer.provenance) == 1
        prov = answer.provenance[0]
        assert prov.source_id == "f1"
        assert prov.metadata["file_path"] == "src/mod.py"
        assert prov.metadata["line_range"] == [3, 5]
        assert prov.metadata["kind"] == "function"  # address.metadata overrides with specific kind

    def test_context_type_results_excluded_from_provenance(self, mock_chat, config):
        synth = CodeSynthesizer(mock_chat, config)
        results = [
            _make_result(),
            _make_result(context_type="imports"),
        ]
        answer = synth.generate("test", "context", results)

        # Only the non-context-type result should be in provenance
        assert len(answer.provenance) == 1

    def test_strict_grounding_prompt(self, mock_chat, config):
        synth = CodeSynthesizer(mock_chat, config)
        synth.generate("test", "context", [_make_result()])

        call_args = mock_chat.chat.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "ONLY" in system_msg

    def test_open_grounding_prompt(self, mock_chat):
        config = FitzKragConfig(collection="test", strict_grounding=False)
        synth = CodeSynthesizer(mock_chat, config)
        synth.generate("test", "context", [_make_result()])

        call_args = mock_chat.chat.call_args[0][0]
        system_msg = call_args[0]["content"]
        assert "supplement" in system_msg

    def test_llm_failure_raises(self, config):
        chat = MagicMock()
        chat.chat.side_effect = Exception("API error")
        synth = CodeSynthesizer(chat, config)

        with pytest.raises(Exception, match="API error"):
            synth.generate("test", "context", [_make_result()])
