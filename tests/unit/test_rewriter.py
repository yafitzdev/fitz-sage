# tests/unit/test_rewriter.py
"""
Tests for fitz_ai.retrieval.rewriter module.

Tests cover:
1. QueryRewriter - query rewriting with LLM
2. ConversationContext - conversation history handling
3. RewriteResult - rewrite result handling
4. Response parsing - JSON extraction and fallback handling
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from fitz_ai.retrieval.rewriter import (
    ConversationContext,
    ConversationMessage,
    QueryRewriter,
    RewriteResult,
    RewriteType,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockChatClient:
    """Mock chat client for testing."""

    response: str = '{"rewritten_query": "test query", "rewrite_type": "none", "confidence": 1.0, "is_ambiguous": false, "disambiguated_queries": []}'
    calls: list = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.response


# ---------------------------------------------------------------------------
# Tests for ConversationContext
# ---------------------------------------------------------------------------


class TestConversationContext:
    """Tests for ConversationContext class."""

    def test_empty_context(self):
        """Test empty conversation context."""
        context = ConversationContext()
        assert context.is_empty()
        assert context.format_for_prompt() == ""

    def test_context_with_history(self):
        """Test context with conversation history."""
        context = ConversationContext(
            history=[
                ConversationMessage(role="user", content="Hello"),
                ConversationMessage(role="assistant", content="Hi there!"),
            ]
        )
        assert not context.is_empty()
        formatted = context.format_for_prompt()
        assert "User: Hello" in formatted
        assert "Assistant: Hi there!" in formatted

    def test_recent_history_limit(self):
        """Test that recent history is limited by max_turns."""
        messages = []
        for i in range(20):
            messages.append(ConversationMessage(role="user", content=f"Message {i}"))
            messages.append(ConversationMessage(role="assistant", content=f"Reply {i}"))

        context = ConversationContext(history=messages, max_turns=3)
        recent = context.recent_history()

        # Should only have last 6 messages (3 turns * 2 messages)
        assert len(recent) == 6

    def test_format_truncates_long_messages(self):
        """Test that long messages are truncated in formatting."""
        long_content = "A" * 600  # Longer than 500 char limit
        context = ConversationContext(
            history=[ConversationMessage(role="user", content=long_content)]
        )
        formatted = context.format_for_prompt()
        assert "..." in formatted
        assert len(formatted) < len(long_content)


# ---------------------------------------------------------------------------
# Tests for RewriteResult
# ---------------------------------------------------------------------------


class TestRewriteResult:
    """Tests for RewriteResult class."""

    def test_was_rewritten_none(self):
        """Test was_rewritten returns False for NONE type."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="test",
            rewrite_type=RewriteType.NONE,
            confidence=1.0,
        )
        assert not result.was_rewritten

    def test_was_rewritten_same_query(self):
        """Test was_rewritten returns False when queries are identical."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="test",
            rewrite_type=RewriteType.CLARITY,
            confidence=1.0,
        )
        assert not result.was_rewritten

    def test_was_rewritten_different_query(self):
        """Test was_rewritten returns True when query changed."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="improved test",
            rewrite_type=RewriteType.CLARITY,
            confidence=1.0,
        )
        assert result.was_rewritten

    def test_all_query_variations_no_rewrite(self):
        """Test query variations with no rewrite."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="test",
            rewrite_type=RewriteType.NONE,
            confidence=1.0,
        )
        variations = result.all_query_variations
        assert variations == ["test"]

    def test_all_query_variations_with_rewrite(self):
        """Test query variations with rewrite."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="improved test",
            rewrite_type=RewriteType.CLARITY,
            confidence=1.0,
        )
        variations = result.all_query_variations
        assert "test" in variations
        assert "improved test" in variations
        assert len(variations) == 2

    def test_all_query_variations_with_disambiguated(self):
        """Test query variations with disambiguated queries."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="improved test",
            rewrite_type=RewriteType.COMBINED,
            confidence=0.8,
            is_ambiguous=True,
            disambiguated_queries=["interpretation 1", "interpretation 2"],
        )
        variations = result.all_query_variations
        assert len(variations) == 4
        assert "test" in variations
        assert "improved test" in variations
        assert "interpretation 1" in variations
        assert "interpretation 2" in variations


# ---------------------------------------------------------------------------
# Tests for QueryRewriter
# ---------------------------------------------------------------------------


class TestQueryRewriter:
    """Tests for QueryRewriter class."""

    def test_basic_rewrite(self):
        """Test basic query rewriting."""
        response = json.dumps(
            {
                "rewritten_query": "improved query about their products and services",
                "rewrite_type": "clarity",
                "confidence": 0.9,
                "is_ambiguous": False,
                "disambiguated_queries": [],
            }
        )
        mock_chat = MockChatClient(response=response)
        rewriter = QueryRewriter(chat=mock_chat)

        # Use a query with pronoun to trigger rewriting heuristic
        result = rewriter.rewrite("Tell me about their products")

        assert result.original_query == "Tell me about their products"
        assert result.rewritten_query == "improved query about their products and services"
        assert result.rewrite_type == RewriteType.CLARITY
        assert result.confidence == 0.9

    def test_chat_called_with_prompt(self):
        """Test that chat is called with correct message structure."""
        mock_chat = MockChatClient()
        rewriter = QueryRewriter(chat=mock_chat)

        # Use a query with pronoun to trigger rewriting heuristic
        rewriter.rewrite("What does it do and how does it work")

        assert len(mock_chat.calls) == 1
        messages = mock_chat.calls[0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "What does it do" in messages[0]["content"]

    def test_short_query_skipped(self):
        """Test that very short queries skip rewriting."""
        mock_chat = MockChatClient()
        rewriter = QueryRewriter(chat=mock_chat, min_query_length=5)

        result = rewriter.rewrite("ab")

        assert len(mock_chat.calls) == 0
        assert result.rewrite_type == RewriteType.NONE
        assert result.rewritten_query == "ab"

    def test_conversation_context_included(self):
        """Test that conversation context is included in prompt."""
        mock_chat = MockChatClient()
        rewriter = QueryRewriter(chat=mock_chat)
        context = ConversationContext(
            history=[
                ConversationMessage(role="user", content="Tell me about TechCorp"),
                ConversationMessage(role="assistant", content="TechCorp is a company..."),
            ]
        )

        rewriter.rewrite("What are their products?", context)

        prompt = mock_chat.calls[0][0]["content"]
        assert "TechCorp" in prompt
        assert "Conversation History" in prompt

    def test_conversational_rewrite(self):
        """Test conversational pronoun resolution."""
        response = json.dumps(
            {
                "rewritten_query": "What are TechCorp's products?",
                "rewrite_type": "conversational",
                "confidence": 0.95,
                "is_ambiguous": False,
                "disambiguated_queries": [],
            }
        )
        mock_chat = MockChatClient(response=response)
        rewriter = QueryRewriter(chat=mock_chat)
        context = ConversationContext(
            history=[
                ConversationMessage(role="user", content="Tell me about TechCorp"),
            ]
        )

        result = rewriter.rewrite("What are their products?", context)

        assert result.rewrite_type == RewriteType.CONVERSATIONAL
        assert "TechCorp" in result.rewritten_query

    def test_ambiguous_query_detection(self):
        """Test ambiguous query detection."""
        response = json.dumps(
            {
                "rewritten_query": "How to use Python programming and also set up the environment",
                "rewrite_type": "combined",
                "confidence": 0.7,
                "is_ambiguous": True,
                "disambiguated_queries": [
                    "Python programming language basics",
                    "Python programming setup guide",
                ],
            }
        )
        mock_chat = MockChatClient(response=response)
        rewriter = QueryRewriter(chat=mock_chat)

        # Use a compound query to trigger rewriting heuristic
        result = rewriter.rewrite("How do I use Python and also set up the environment?")

        assert result.is_ambiguous
        assert len(result.disambiguated_queries) == 2

    def test_chat_error_returns_original(self):
        """Test that chat errors return original query."""

        class ErrorChat:
            def chat(self, messages):
                raise RuntimeError("API error")

        rewriter = QueryRewriter(chat=ErrorChat())

        # Use query with pronoun to trigger heuristic (force LLM call)
        query = "What does their system do and how does it work?"
        result = rewriter.rewrite(query)

        assert result.rewrite_type == RewriteType.NONE
        assert result.rewritten_query == query
        assert result.confidence == 0.0

    def test_no_rewrite_needed(self):
        """Test when no rewrite is needed (LLM says query is already good)."""
        # Query has pronoun to trigger heuristic, but LLM says no rewrite needed
        query = "What is their revenue for the fiscal year and also operating costs?"
        response = json.dumps(
            {
                "rewritten_query": query,  # Same as original
                "rewrite_type": "none",
                "confidence": 1.0,
                "is_ambiguous": False,
                "disambiguated_queries": [],
            }
        )
        mock_chat = MockChatClient(response=response)
        rewriter = QueryRewriter(chat=mock_chat)

        result = rewriter.rewrite(query)

        assert result.rewrite_type == RewriteType.NONE
        assert not result.was_rewritten


# ---------------------------------------------------------------------------
# Tests for Response Parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    """Tests for response parsing logic."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = json.dumps(
            {
                "rewritten_query": "test about their functionality",
                "rewrite_type": "clarity",
                "confidence": 0.9,
                "is_ambiguous": False,
                "disambiguated_queries": [],
            }
        )
        mock_chat = MockChatClient(response=response)
        rewriter = QueryRewriter(chat=mock_chat)

        # Use query with pronoun to trigger heuristic
        result = rewriter.rewrite("Explain their functionality")

        assert result.rewritten_query == "test about their functionality"
        assert result.rewrite_type == RewriteType.CLARITY

    def test_parse_json_in_markdown(self):
        """Test parsing JSON wrapped in markdown code block."""
        response = """```json
{
    "rewritten_query": "test about their implementation",
    "rewrite_type": "retrieval",
    "confidence": 0.8,
    "is_ambiguous": false,
    "disambiguated_queries": []
}
```"""
        mock_chat = MockChatClient(response=response)
        rewriter = QueryRewriter(chat=mock_chat)

        # Use query with pronoun to trigger heuristic
        result = rewriter.rewrite("Describe their implementation")

        assert result.rewritten_query == "test about their implementation"
        assert result.rewrite_type == RewriteType.RETRIEVAL

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON embedded in response text."""
        response = """Here's the rewritten query:
{"rewritten_query": "clarified query about their system", "rewrite_type": "clarity", "confidence": 0.9, "is_ambiguous": false, "disambiguated_queries": []}
That should work better."""
        mock_chat = MockChatClient(response=response)
        rewriter = QueryRewriter(chat=mock_chat)

        # Use query with pronoun to trigger heuristic
        result = rewriter.rewrite("Explain their system")

        assert result.rewritten_query == "clarified query about their system"

    def test_parse_invalid_json_returns_original(self):
        """Test that invalid JSON returns original query."""
        response = "This is not valid JSON"
        mock_chat = MockChatClient(response=response)
        rewriter = QueryRewriter(chat=mock_chat)

        # Use query with pronoun to trigger heuristic
        query = "Explain their original query processing"
        result = rewriter.rewrite(query)

        assert result.rewritten_query == query
        assert result.rewrite_type == RewriteType.NONE

    def test_parse_empty_rewritten_query(self):
        """Test that empty rewritten query falls back to original."""
        response = json.dumps(
            {
                "rewritten_query": "",
                "rewrite_type": "clarity",
                "confidence": 0.9,
                "is_ambiguous": False,
                "disambiguated_queries": [],
            }
        )
        mock_chat = MockChatClient(response=response)
        rewriter = QueryRewriter(chat=mock_chat)

        # Use query with pronoun to trigger heuristic
        query = "Describe their original query processing"
        result = rewriter.rewrite(query)

        assert result.rewritten_query == query
        assert result.rewrite_type == RewriteType.NONE

    def test_parse_limits_disambiguated_queries(self):
        """Test that disambiguated queries are limited to 3."""
        response = json.dumps(
            {
                "rewritten_query": "test about their features",
                "rewrite_type": "combined",
                "confidence": 0.7,
                "is_ambiguous": True,
                "disambiguated_queries": ["q1", "q2", "q3", "q4", "q5"],
            }
        )
        mock_chat = MockChatClient(response=response)
        rewriter = QueryRewriter(chat=mock_chat)

        # Use query with pronoun to trigger heuristic
        result = rewriter.rewrite("Describe their features")

        assert len(result.disambiguated_queries) == 3


# ---------------------------------------------------------------------------
# Tests for Prompt Loading
# ---------------------------------------------------------------------------


class TestPromptLoading:
    """Tests for prompt template loading."""

    def test_default_prompt_loaded(self):
        """Test that default prompt template is loaded."""
        mock_chat = MockChatClient()
        rewriter = QueryRewriter(chat=mock_chat)

        assert rewriter.prompt_template is not None
        assert len(rewriter.prompt_template) > 0

    def test_prompt_has_placeholders(self):
        """Test that default prompt has required placeholders."""
        mock_chat = MockChatClient()
        rewriter = QueryRewriter(chat=mock_chat)

        assert "{query}" in rewriter.prompt_template
        assert "{history_section}" in rewriter.prompt_template

    def test_custom_prompt_template(self):
        """Test custom prompt template."""
        mock_chat = MockChatClient()
        custom_template = "Rewrite: {query}\n{history_section}"
        rewriter = QueryRewriter(chat=mock_chat, prompt_template=custom_template)

        # Use query with pronoun to trigger heuristic
        rewriter.rewrite("Explain their test functionality")

        prompt = mock_chat.calls[0][0]["content"]
        assert "Rewrite: Explain their test functionality" in prompt
