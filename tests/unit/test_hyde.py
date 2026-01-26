# tests/unit/test_hyde.py
"""
Tests for fitz_ai.retrieval.hyde module.

Tests cover:
1. HydeGenerator - hypothesis generation from queries
2. Response parsing - JSON extraction and fallback handling
3. Prompt building - template interpolation
"""

from __future__ import annotations

from dataclasses import dataclass

from fitz_ai.retrieval.hyde import HydeGenerator

# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockChatClient:
    """Mock chat client for testing."""

    response: str = '["Passage 1", "Passage 2", "Passage 3"]'
    calls: list = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.response


def create_mock_chat_factory(mock_chat):
    """Create a mock chat factory that returns the mock chat client."""

    def factory(tier: str = "fast"):
        return mock_chat

    return factory


# ---------------------------------------------------------------------------
# Tests for HydeGenerator
# ---------------------------------------------------------------------------


class TestHydeGenerator:
    """Tests for HydeGenerator class."""

    def test_basic_generation(self):
        """Test basic hypothesis generation."""
        mock_chat = MockChatClient()
        generator = HydeGenerator(chat_factory=create_mock_chat_factory(mock_chat))

        hypotheses = generator.generate("What is the refund policy?")

        assert len(hypotheses) == 3
        assert hypotheses[0] == "Passage 1"
        assert hypotheses[1] == "Passage 2"
        assert hypotheses[2] == "Passage 3"

    def test_chat_called_with_prompt(self):
        """Test that chat is called with correct message structure."""
        mock_chat = MockChatClient()
        generator = HydeGenerator(chat_factory=create_mock_chat_factory(mock_chat))

        generator.generate("What is the refund policy?")

        assert len(mock_chat.calls) == 1
        messages = mock_chat.calls[0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "What is the refund policy?" in messages[0]["content"]

    def test_prompt_contains_query(self):
        """Test that prompt includes the query."""
        mock_chat = MockChatClient()
        generator = HydeGenerator(chat_factory=create_mock_chat_factory(mock_chat))

        generator.generate("My specific question here")

        prompt = mock_chat.calls[0][0]["content"]
        assert "My specific question here" in prompt

    def test_prompt_contains_num_hypotheses(self):
        """Test that prompt includes hypothesis count."""
        mock_chat = MockChatClient()
        generator = HydeGenerator(
            chat_factory=create_mock_chat_factory(mock_chat), num_hypotheses=5
        )

        generator.generate("Test query")

        prompt = mock_chat.calls[0][0]["content"]
        assert "5" in prompt

    def test_custom_num_hypotheses(self):
        """Test custom number of hypotheses."""
        mock_chat = MockChatClient(response='["One", "Two"]')
        generator = HydeGenerator(
            chat_factory=create_mock_chat_factory(mock_chat), num_hypotheses=2
        )

        hypotheses = generator.generate("Test")

        assert len(hypotheses) == 2

    def test_limits_to_num_hypotheses(self):
        """Test that results are limited to num_hypotheses."""
        mock_chat = MockChatClient(response='["P1", "P2", "P3", "P4", "P5", "P6"]')
        generator = HydeGenerator(
            chat_factory=create_mock_chat_factory(mock_chat), num_hypotheses=3
        )

        hypotheses = generator.generate("Test")

        assert len(hypotheses) == 3

    def test_custom_prompt_template(self):
        """Test custom prompt template."""
        mock_chat = MockChatClient()
        custom_template = "Custom template: {query} with {num_hypotheses} results"
        generator = HydeGenerator(
            chat_factory=create_mock_chat_factory(mock_chat),
            prompt_template=custom_template,
        )

        generator.generate("My query")

        prompt = mock_chat.calls[0][0]["content"]
        assert "Custom template: My query with 3 results" == prompt

    def test_empty_response_returns_empty_list(self):
        """Test that empty response returns empty list."""
        mock_chat = MockChatClient(response="[]")
        generator = HydeGenerator(chat_factory=create_mock_chat_factory(mock_chat))

        hypotheses = generator.generate("Test")

        assert hypotheses == []

    def test_chat_error_returns_empty_list(self):
        """Test that chat errors return empty list."""

        class ErrorChat:
            def chat(self, messages):
                raise RuntimeError("API error")

        generator = HydeGenerator(chat_factory=create_mock_chat_factory(ErrorChat()))

        hypotheses = generator.generate("Test")

        assert hypotheses == []


class TestResponseParsing:
    """Tests for response parsing logic."""

    def test_parse_valid_json_array(self):
        """Test parsing valid JSON array response."""
        mock_chat = MockChatClient(
            response='["First passage about topic", "Second passage about topic"]'
        )
        generator = HydeGenerator(
            chat_factory=create_mock_chat_factory(mock_chat), num_hypotheses=2
        )

        hypotheses = generator.generate("Test")

        assert len(hypotheses) == 2
        assert hypotheses[0] == "First passage about topic"
        assert hypotheses[1] == "Second passage about topic"

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON array embedded in response text."""
        mock_chat = MockChatClient(
            response='Here are the passages:\n["Passage 1", "Passage 2"]\n\nEnd of response.'
        )
        generator = HydeGenerator(
            chat_factory=create_mock_chat_factory(mock_chat), num_hypotheses=2
        )

        hypotheses = generator.generate("Test")

        assert len(hypotheses) == 2

    def test_parse_filters_empty_strings(self):
        """Test that empty strings are filtered from results."""
        mock_chat = MockChatClient(response='["Valid passage", "", "Another valid"]')
        generator = HydeGenerator(
            chat_factory=create_mock_chat_factory(mock_chat), num_hypotheses=3
        )

        hypotheses = generator.generate("Test")

        assert len(hypotheses) == 2
        assert "" not in hypotheses

    def test_parse_strips_whitespace(self):
        """Test that whitespace is stripped from passages."""
        mock_chat = MockChatClient(
            response='["  Passage with spaces  ", "\\nPassage with newlines\\n"]'
        )
        generator = HydeGenerator(
            chat_factory=create_mock_chat_factory(mock_chat), num_hypotheses=2
        )

        hypotheses = generator.generate("Test")

        assert hypotheses[0] == "Passage with spaces"

    def test_parse_invalid_json_falls_back(self):
        """Test fallback parsing when JSON is invalid."""
        mock_chat = MockChatClient(
            response="""Here are some hypothetical passages:

This is the first passage about the topic that provides useful context.
This is the second passage with different perspective on the matter."""
        )
        generator = HydeGenerator(
            chat_factory=create_mock_chat_factory(mock_chat), num_hypotheses=2
        )

        hypotheses = generator.generate("Test")

        # Should get at least some content from fallback
        assert len(hypotheses) >= 1

    def test_parse_handles_non_string_items(self):
        """Test that non-string items in array are converted."""
        mock_chat = MockChatClient(response='["Valid string", 123, true, {"key": "value"}]')
        generator = HydeGenerator(
            chat_factory=create_mock_chat_factory(mock_chat), num_hypotheses=4
        )

        hypotheses = generator.generate("Test")

        # Only valid strings should be included
        assert len(hypotheses) == 1
        assert hypotheses[0] == "Valid string"


class TestPromptLoading:
    """Tests for prompt template loading."""

    def test_default_prompt_loaded(self):
        """Test that default prompt template is loaded."""
        mock_chat = MockChatClient()
        generator = HydeGenerator(chat_factory=create_mock_chat_factory(mock_chat))

        # Should have loaded the default template
        assert generator.prompt_template is not None
        assert len(generator.prompt_template) > 0

    def test_prompt_has_placeholders(self):
        """Test that default prompt has required placeholders."""
        mock_chat = MockChatClient()
        generator = HydeGenerator(chat_factory=create_mock_chat_factory(mock_chat))

        assert "{query}" in generator.prompt_template
        assert "{num_hypotheses}" in generator.prompt_template
