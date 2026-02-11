# tests/unit/test_query_analyzer.py
"""Tests for QueryAnalyzer — LLM-based query classification."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from fitz_ai.engines.fitz_krag.query_analyzer import (
    QueryAnalysis,
    QueryAnalyzer,
    QueryType,
    _parse_query_type,
)


@pytest.fixture
def mock_chat():
    return MagicMock()


@pytest.fixture
def analyzer(mock_chat):
    return QueryAnalyzer(mock_chat)


def _mock_response(data: dict) -> str:
    return json.dumps(data)


class TestQueryType:
    def test_values(self):
        assert QueryType.CODE.value == "code"
        assert QueryType.DOCUMENTATION.value == "documentation"
        assert QueryType.GENERAL.value == "general"
        assert QueryType.CROSS.value == "cross"
        assert QueryType.DATA.value == "data"

    def test_parse_valid(self):
        assert _parse_query_type("code") == QueryType.CODE
        assert _parse_query_type("documentation") == QueryType.DOCUMENTATION
        assert _parse_query_type("CODE") == QueryType.CODE

    def test_parse_invalid_defaults_general(self):
        assert _parse_query_type("unknown") == QueryType.GENERAL
        assert _parse_query_type("") == QueryType.GENERAL


class TestQueryAnalysis:
    def test_strategy_weights_code(self):
        analysis = QueryAnalysis(primary_type=QueryType.CODE)
        weights = analysis.strategy_weights
        assert weights["code"] == 0.75
        assert weights["section"] == 0.1
        assert weights["table"] == 0.05
        assert weights["chunk"] == 0.1

    def test_strategy_weights_documentation(self):
        analysis = QueryAnalysis(primary_type=QueryType.DOCUMENTATION)
        weights = analysis.strategy_weights
        assert weights["section"] == 0.75
        assert weights["code"] == 0.1
        assert weights["table"] == 0.05

    def test_strategy_weights_cross(self):
        analysis = QueryAnalysis(primary_type=QueryType.CROSS)
        weights = analysis.strategy_weights
        assert weights["code"] == 0.35
        assert weights["section"] == 0.35
        assert weights["table"] == 0.1

    def test_strategy_weights_general(self):
        analysis = QueryAnalysis(primary_type=QueryType.GENERAL)
        weights = analysis.strategy_weights
        assert weights["chunk"] == 0.35
        assert weights["table"] == 0.15

    def test_strategy_weights_data(self):
        analysis = QueryAnalysis(primary_type=QueryType.DATA)
        weights = analysis.strategy_weights
        assert weights["table"] == 0.85
        assert weights["code"] == 0.05
        assert weights["section"] == 0.05
        assert weights["chunk"] == 0.05

    def test_frozen(self):
        analysis = QueryAnalysis(primary_type=QueryType.CODE)
        with pytest.raises(AttributeError):
            analysis.primary_type = QueryType.GENERAL


class TestQueryAnalyzer:
    def test_classifies_code_query(self, analyzer, mock_chat):
        mock_chat.chat.return_value = _mock_response(
            {
                "primary_type": "code",
                "secondary_type": None,
                "confidence": 0.9,
                "entities": ["authenticate", "UserService"],
                "refined_query": "how does the authenticate function work",
            }
        )
        result = analyzer.analyze("how does authenticate work?")
        assert result.primary_type == QueryType.CODE
        assert result.confidence == 0.9
        assert "authenticate" in result.entities
        assert "UserService" in result.entities

    def test_classifies_documentation_query(self, analyzer, mock_chat):
        mock_chat.chat.return_value = _mock_response(
            {
                "primary_type": "documentation",
                "confidence": 0.85,
                "entities": ["Results"],
                "refined_query": "what does the Results section say",
            }
        )
        result = analyzer.analyze("what does the Results section say?")
        assert result.primary_type == QueryType.DOCUMENTATION

    def test_classifies_cross_query(self, analyzer, mock_chat):
        mock_chat.chat.return_value = _mock_response(
            {
                "primary_type": "cross",
                "secondary_type": "code",
                "confidence": 0.75,
                "entities": ["authentication"],
                "refined_query": "how does code implement the spec's auth flow",
            }
        )
        result = analyzer.analyze("how does the code implement the spec's auth flow?")
        assert result.primary_type == QueryType.CROSS
        assert result.secondary_type == QueryType.CODE

    def test_handles_llm_failure(self, analyzer, mock_chat):
        mock_chat.chat.side_effect = RuntimeError("LLM error")
        result = analyzer.analyze("some query")
        assert result.primary_type == QueryType.GENERAL
        assert result.confidence == 0.3

    def test_handles_invalid_json(self, analyzer, mock_chat):
        mock_chat.chat.return_value = "not valid json {{"
        result = analyzer.analyze("some query")
        assert result.primary_type == QueryType.GENERAL
        assert result.confidence == 0.3

    def test_handles_markdown_wrapped_json(self, analyzer, mock_chat):
        data = {
            "primary_type": "code",
            "confidence": 0.8,
            "entities": [],
            "refined_query": "test",
        }
        mock_chat.chat.return_value = f"```json\n{json.dumps(data)}\n```"
        result = analyzer.analyze("test")
        assert result.primary_type == QueryType.CODE

    def test_clamps_confidence(self, analyzer, mock_chat):
        mock_chat.chat.return_value = _mock_response(
            {
                "primary_type": "code",
                "confidence": 5.0,
                "entities": [],
                "refined_query": "test",
            }
        )
        result = analyzer.analyze("test")
        assert result.confidence == 1.0

    def test_handles_missing_fields(self, analyzer, mock_chat):
        mock_chat.chat.return_value = _mock_response({"primary_type": "code"})
        result = analyzer.analyze("test")
        assert result.primary_type == QueryType.CODE
        assert result.entities == ()

    def test_classifies_data_query(self, analyzer, mock_chat):
        mock_chat.chat.return_value = _mock_response(
            {
                "primary_type": "data",
                "confidence": 0.9,
                "entities": ["salary"],
                "refined_query": "average salary from the dataset",
            }
        )
        result = analyzer.analyze("what's the average salary?")
        assert result.primary_type == QueryType.DATA
        assert result.confidence == 0.9

    def test_data_type_parsed(self):
        assert _parse_query_type("data") == QueryType.DATA
