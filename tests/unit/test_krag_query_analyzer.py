# tests/unit/test_krag_query_analyzer.py
"""Unit tests for fitz_krag query analyzer module."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from fitz_ai.engines.fitz_krag.query_analyzer import (
    QueryAnalysis,
    QueryAnalyzer,
    QueryType,
    _parse_query_type,
)

# ---------------------------------------------------------------------------
# TestQueryType
# ---------------------------------------------------------------------------


class TestQueryType:
    """Tests for the QueryType enum."""

    def test_query_type_values(self) -> None:
        """Enum values match expected strings."""
        assert QueryType.CODE.value == "code"
        assert QueryType.DOCUMENTATION.value == "documentation"
        assert QueryType.GENERAL.value == "general"
        assert QueryType.CROSS.value == "cross"

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("code", QueryType.CODE),
            ("documentation", QueryType.DOCUMENTATION),
            ("general", QueryType.GENERAL),
            ("cross", QueryType.CROSS),
        ],
    )
    def test_parse_query_type_valid(self, value: str, expected: QueryType) -> None:
        """Each valid lowercase value maps to the correct enum member."""
        assert _parse_query_type(value) is expected

    def test_parse_query_type_invalid(self) -> None:
        """Unknown string falls back to GENERAL."""
        assert _parse_query_type("nonexistent") is QueryType.GENERAL
        assert _parse_query_type("") is QueryType.GENERAL
        assert _parse_query_type("foo_bar") is QueryType.GENERAL

    @pytest.mark.parametrize("raw", ["CODE", "Code", "CoDe", "DOCUMENTATION"])
    def test_parse_query_type_case(self, raw: str) -> None:
        """Parsing is case-insensitive (value.lower() inside _parse_query_type)."""
        result = _parse_query_type(raw)
        assert result is QueryType(raw.lower())


# ---------------------------------------------------------------------------
# TestQueryAnalysis
# ---------------------------------------------------------------------------


class TestQueryAnalysis:
    """Tests for the QueryAnalysis dataclass and strategy_weights property."""

    def test_strategy_weights_code(self) -> None:
        """CODE type produces code-heavy weights."""
        analysis = QueryAnalysis(primary_type=QueryType.CODE)
        assert analysis.strategy_weights == {
            "code": 0.8,
            "section": 0.1,
            "chunk": 0.1,
        }

    def test_strategy_weights_documentation(self) -> None:
        """DOCUMENTATION type produces section-heavy weights."""
        analysis = QueryAnalysis(primary_type=QueryType.DOCUMENTATION)
        assert analysis.strategy_weights == {
            "code": 0.1,
            "section": 0.8,
            "chunk": 0.1,
        }

    def test_strategy_weights_general(self) -> None:
        """GENERAL type produces balanced weights."""
        analysis = QueryAnalysis(primary_type=QueryType.GENERAL)
        assert analysis.strategy_weights == {
            "code": 0.3,
            "section": 0.3,
            "chunk": 0.4,
        }

    def test_strategy_weights_cross(self) -> None:
        """CROSS type produces 0.4/0.4/0.2 weights."""
        analysis = QueryAnalysis(primary_type=QueryType.CROSS)
        assert analysis.strategy_weights == {
            "code": 0.4,
            "section": 0.4,
            "chunk": 0.2,
        }

    def test_strategy_weights_returns_copy(self) -> None:
        """strategy_weights returns a new dict each call (not the internal one)."""
        analysis = QueryAnalysis(primary_type=QueryType.CODE)
        w1 = analysis.strategy_weights
        w2 = analysis.strategy_weights
        assert w1 == w2
        assert w1 is not w2

    def test_frozen(self) -> None:
        """QueryAnalysis is frozen -- fields cannot be reassigned."""
        analysis = QueryAnalysis(primary_type=QueryType.CODE, confidence=0.9)
        with pytest.raises(FrozenInstanceError):
            analysis.primary_type = QueryType.GENERAL  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            analysis.confidence = 0.1  # type: ignore[misc]

    def test_defaults(self) -> None:
        """Default field values are sane."""
        analysis = QueryAnalysis(primary_type=QueryType.GENERAL)
        assert analysis.secondary_type is None
        assert analysis.confidence == 0.5
        assert analysis.entities == ()
        assert analysis.refined_query == ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_mock(response: str) -> MagicMock:
    """Create a MagicMock ChatProvider that returns *response* from chat()."""
    mock = MagicMock()
    mock.chat.return_value = response
    return mock


def _json_response(
    primary: str = "code",
    secondary: str | None = None,
    confidence: float = 0.85,
    entities: list[str] | None = None,
    refined: str = "refined query",
) -> str:
    """Build a valid JSON string matching the LLM response schema."""
    payload: dict = {
        "primary_type": primary,
        "secondary_type": secondary,
        "confidence": confidence,
        "entities": entities or [],
        "refined_query": refined,
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# TestQueryAnalyzer
# ---------------------------------------------------------------------------


class TestQueryAnalyzer:
    """Tests for QueryAnalyzer.analyze() and its internal _parse_response."""

    def test_analyze_code_query(self) -> None:
        """LLM returns a code classification -- parsed correctly."""
        resp = _json_response(primary="code", confidence=0.9, refined="find the parse function")
        chat = _make_chat_mock(resp)
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("where is the parse function?")

        assert result.primary_type is QueryType.CODE
        assert result.confidence == pytest.approx(0.9)
        assert result.refined_query == "find the parse function"
        chat.chat.assert_called_once()

    def test_analyze_documentation_query(self) -> None:
        """LLM returns documentation classification."""
        resp = _json_response(
            primary="documentation",
            confidence=0.8,
            refined="deployment procedures",
        )
        chat = _make_chat_mock(resp)
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("show me the deployment docs")

        assert result.primary_type is QueryType.DOCUMENTATION
        assert result.confidence == pytest.approx(0.8)

    def test_analyze_with_entities(self) -> None:
        """Entities list from the LLM JSON is captured as a tuple."""
        resp = _json_response(
            primary="code",
            entities=["QueryAnalyzer", "parse_response"],
            refined="QueryAnalyzer.parse_response",
        )
        chat = _make_chat_mock(resp)
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("how does QueryAnalyzer parse responses?")

        assert result.entities == ("QueryAnalyzer", "parse_response")

    def test_analyze_with_secondary_type(self) -> None:
        """secondary_type is parsed when the LLM provides one."""
        resp = _json_response(primary="cross", secondary="code")
        chat = _make_chat_mock(resp)
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("compare code and docs")

        assert result.primary_type is QueryType.CROSS
        assert result.secondary_type is QueryType.CODE

    def test_analyze_llm_failure_falls_back(self) -> None:
        """When chat() raises an exception, fall back to GENERAL at 0.3."""
        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("LLM unavailable")
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("anything at all")

        assert result.primary_type is QueryType.GENERAL
        assert result.confidence == pytest.approx(0.3)
        assert result.refined_query == "anything at all"

    def test_analyze_invalid_json_falls_back(self) -> None:
        """Non-JSON LLM output falls back to GENERAL at 0.3."""
        chat = _make_chat_mock("I'm not sure, here is some text.")
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("test query")

        assert result.primary_type is QueryType.GENERAL
        assert result.confidence == pytest.approx(0.3)
        assert result.refined_query == "test query"

    def test_analyze_markdown_wrapped_json(self) -> None:
        """LLM wraps response in ```json ... ``` fences -- still parses."""
        inner = _json_response(primary="documentation", confidence=0.75)
        wrapped = f"```json\n{inner}\n```"
        chat = _make_chat_mock(wrapped)
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("what are the API docs?")

        assert result.primary_type is QueryType.DOCUMENTATION
        assert result.confidence == pytest.approx(0.75)

    def test_analyze_markdown_bare_backtick_fence(self) -> None:
        """LLM wraps response in bare ``` fences (no language tag)."""
        inner = _json_response(primary="code", confidence=0.6)
        wrapped = f"```\n{inner}\n```"
        chat = _make_chat_mock(wrapped)
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("find function foo")

        assert result.primary_type is QueryType.CODE
        assert result.confidence == pytest.approx(0.6)

    def test_analyze_confidence_clamped_high(self) -> None:
        """Confidence > 1.0 is clamped to 1.0."""
        resp = _json_response(primary="code", confidence=5.0)
        chat = _make_chat_mock(resp)
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("query")

        assert result.confidence == pytest.approx(1.0)

    def test_analyze_confidence_clamped_low(self) -> None:
        """Confidence < 0.0 is clamped to 0.0."""
        resp = _json_response(primary="code", confidence=-2.0)
        chat = _make_chat_mock(resp)
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("query")

        assert result.confidence == pytest.approx(0.0)

    def test_analyze_missing_fields_defaults(self) -> None:
        """Partial JSON fills missing fields with defaults."""
        chat = _make_chat_mock('{"primary_type": "general"}')
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("vague question")

        assert result.primary_type is QueryType.GENERAL
        assert result.secondary_type is None
        assert result.confidence == pytest.approx(0.5)
        assert result.entities == ()
        assert result.refined_query == "vague question"

    def test_analyze_entities_non_list_ignored(self) -> None:
        """If entities is not a list in the JSON, it falls back to empty."""
        payload = json.dumps(
            {
                "primary_type": "code",
                "confidence": 0.8,
                "entities": "not a list",
                "refined_query": "q",
            }
        )
        chat = _make_chat_mock(payload)
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("query")

        assert result.entities == ()

    def test_analyze_unknown_primary_type_defaults(self) -> None:
        """Unknown primary_type string in JSON falls back to GENERAL."""
        payload = json.dumps(
            {
                "primary_type": "unknown_type",
                "confidence": 0.7,
            }
        )
        chat = _make_chat_mock(payload)
        analyzer = QueryAnalyzer(chat=chat)

        result = analyzer.analyze("query")

        assert result.primary_type is QueryType.GENERAL

    def test_analyze_prompt_contains_query(self) -> None:
        """The prompt sent to the LLM includes the original query text."""
        chat = _make_chat_mock(_json_response())
        analyzer = QueryAnalyzer(chat=chat)

        analyzer.analyze("my specific question")

        args = chat.chat.call_args
        messages = args[0][0]  # first positional arg
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "my specific question" in messages[0]["content"]
