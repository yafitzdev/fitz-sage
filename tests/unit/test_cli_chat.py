# tests/test_cli_chat.py
"""
Tests for the chat command.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_ai.cli.cli import app

runner = CliRunner()


class TestChatCommand:
    """Tests for fitz chat command."""

    def test_chat_shows_help(self):
        """Test that chat --help works."""
        result = runner.invoke(app, ["chat", "--help"])

        assert result.exit_code == 0
        assert "chat" in result.output.lower()
        assert "collection" in result.output.lower()

    def test_chat_exits_when_no_collections(self):
        """Test that chat exits with error when no collections found."""
        mock_registry = MagicMock()
        mock_registry.get_list_collections.return_value = []

        with (
            patch("fitz_ai.cli.commands.chat.RICH", False),
            patch("fitz_ai.runtime.get_default_engine", return_value="fitz_krag"),
            patch("fitz_ai.runtime.get_engine_registry", return_value=mock_registry),
        ):
            result = runner.invoke(app, ["chat"])

        assert result.exit_code != 0


class TestChatHelpers:
    """Tests for chat helper functions."""

    def test_cli_context_loads_config(self):
        """Test CLIContext.load() returns raw and typed config."""
        from fitz_ai.cli.context import CLIContext

        mock_ctx = MagicMock()
        mock_ctx.raw_config = {
            "chat": "cohere",
            "embedding": "cohere",
            "vector_db": "pgvector",
            "retrieval_plugin": "dense",
            "collection": "test",
        }
        mock_ctx.typed_config = MagicMock()
        mock_ctx.typed_config.collection = "test"

        with patch("fitz_ai.cli.context.CLIContext.load", return_value=mock_ctx):
            ctx = CLIContext.load(engine="fitz_krag")

        assert ctx.raw_config["chat"] == "cohere"
        assert ctx.typed_config.collection == "test"

    def test_get_collections_returns_sorted_list(self):
        """Test get_collections returns sorted collection list."""
        mock_ctx = MagicMock()
        mock_ctx.get_collections.return_value = ["apple", "middle", "zebra"]

        from fitz_ai.cli.utils import get_collections

        collections = get_collections(mock_ctx)

        assert collections == ["apple", "middle", "zebra"]

    def test_get_collections_handles_error(self):
        """Test get_collections returns empty list on error."""
        mock_ctx = MagicMock()
        mock_ctx.get_collections.return_value = []

        from fitz_ai.cli.utils import get_collections

        collections = get_collections(mock_ctx)

        assert collections == []


class TestBuildMessages:
    """Tests for _build_messages function."""

    def test_build_messages_basic(self):
        """Test _build_messages creates proper message structure."""
        from fitz_ai.cli.commands.chat import _build_messages

        mock_chunk = MagicMock()
        mock_chunk.content = "Chunk content here"

        messages = _build_messages(
            history=[],
            chunks=[mock_chunk],
            current_question="What is RAG?",
        )

        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is RAG?"
        assert "Chunk content" in messages[0]["content"]

    def test_build_messages_with_history(self):
        """Test _build_messages includes conversation history."""
        from fitz_ai.cli.commands.chat import _build_messages

        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]

        messages = _build_messages(
            history=history,
            chunks=[],
            current_question="Follow up?",
        )

        # system + 2 history + 1 current
        assert len(messages) == 4
        assert messages[1]["content"] == "First question"
        assert messages[2]["content"] == "First answer"
        assert messages[3]["content"] == "Follow up?"

    def test_build_messages_trims_history(self):
        """Test _build_messages trims history to MAX_HISTORY_MESSAGES."""
        from fitz_ai.cli.commands.chat import MAX_HISTORY_MESSAGES, _build_messages

        # Create more history than the limit
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"Question {i}"})
            history.append({"role": "assistant", "content": f"Answer {i}"})

        messages = _build_messages(
            history=history,
            chunks=[],
            current_question="Final question",
        )

        # system + trimmed history + current
        assert len(messages) == 1 + MAX_HISTORY_MESSAGES + 1

    def test_build_messages_chunk_numbering(self):
        """Test _build_messages numbers chunks correctly."""
        from fitz_ai.cli.commands.chat import _build_messages

        chunks = []
        for i in range(3):
            chunk = MagicMock()
            chunk.content = f"Content {i}"
            chunks.append(chunk)

        messages = _build_messages(
            history=[],
            chunks=chunks,
            current_question="Question",
        )

        system_content = messages[0]["content"]
        assert "[1]" in system_content
        assert "[2]" in system_content
        assert "[3]" in system_content


class TestChatDisplay:
    """Tests for chat display functions."""

    def test_display_user_message(self, capsys):
        """Test _display_user_message outputs text."""
        with patch("fitz_ai.cli.commands.chat.RICH", False):
            from fitz_ai.cli.commands.chat import _display_user_message

            _display_user_message("Test message")

        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_display_assistant_message(self, capsys):
        """Test _display_assistant_message outputs text."""
        with patch("fitz_ai.cli.commands.chat.RICH", False):
            from fitz_ai.cli.commands.chat import _display_assistant_message

            _display_assistant_message("Assistant response")

        captured = capsys.readouterr()
        assert "Assistant" in captured.out
        assert "response" in captured.out

    def test_display_welcome(self, capsys):
        """Test _display_welcome shows collection name."""
        with patch("fitz_ai.cli.commands.chat.RICH", False):
            from fitz_ai.cli.commands.chat import _display_welcome

            _display_welcome("my_collection")

        captured = capsys.readouterr()
        assert "my_collection" in captured.out

    def test_display_goodbye(self, capsys):
        """Test _display_goodbye shows farewell."""
        with patch("fitz_ai.cli.commands.chat.RICH", False):
            from fitz_ai.cli.commands.chat import _display_goodbye

            _display_goodbye()

        captured = capsys.readouterr()
        assert "goodbye" in captured.out.lower() or "ended" in captured.out.lower()


class TestChatExitCommands:
    """Tests for chat exit handling."""

    def test_chat_exits_on_exit_command(self):
        """Test that chat exits when user types 'exit'."""
        mock_engine = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_list_collections.return_value = ["test"]

        with (
            patch("fitz_ai.cli.commands.chat.RICH", False),
            patch("fitz_ai.runtime.get_default_engine", return_value="fitz_krag"),
            patch("fitz_ai.runtime.get_engine_registry", return_value=mock_registry),
            patch("fitz_ai.runtime.create_engine", return_value=mock_engine),
        ):
            result = runner.invoke(app, ["chat"], input="exit\n")

        assert "goodbye" in result.output.lower() or "ended" in result.output.lower()

    def test_chat_exits_on_quit_command(self):
        """Test that chat exits when user types 'quit'."""
        mock_engine = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_list_collections.return_value = ["test"]

        with (
            patch("fitz_ai.cli.commands.chat.RICH", False),
            patch("fitz_ai.runtime.get_default_engine", return_value="fitz_krag"),
            patch("fitz_ai.runtime.get_engine_registry", return_value=mock_registry),
            patch("fitz_ai.runtime.create_engine", return_value=mock_engine),
        ):
            result = runner.invoke(app, ["chat"], input="quit\n")

        assert "goodbye" in result.output.lower() or "ended" in result.output.lower()
