# tests/unit/cli/test_ingest_direct_text.py
"""
Tests for direct text ingestion in the ingest command.
"""

from fitz_ai.cli.commands.ingest_helpers import is_direct_text as _is_direct_text


class TestIsDirectText:
    """Tests for the _is_direct_text detection function."""

    def test_plain_text_with_spaces(self):
        """Text with spaces should be detected as direct text."""
        assert _is_direct_text("my boss likes red cars") is True

    def test_sentence_with_punctuation(self):
        """Sentences with punctuation should be detected as direct text."""
        assert _is_direct_text("Hello, world! How are you?") is True

    def test_long_text_without_spaces(self):
        """Long text (>100 chars) should be detected as direct text."""
        long_text = "a" * 101
        assert _is_direct_text(long_text) is True

    def test_relative_path(self):
        """Relative paths should not be detected as direct text."""
        assert _is_direct_text("./src") is False
        assert _is_direct_text("../docs") is False

    def test_file_with_extension(self):
        """Files with common extensions should not be detected as direct text."""
        assert _is_direct_text("readme.md") is False
        assert _is_direct_text("data.csv") is False
        assert _is_direct_text("config.yaml") is False
        assert _is_direct_text("test.py") is False

    def test_absolute_paths(self):
        """Absolute paths should not be detected as direct text."""
        assert _is_direct_text("C:/Users/file.txt") is False
        assert _is_direct_text("/home/user/docs") is False

    def test_dotfiles(self):
        """Dotfiles should not be detected as direct text."""
        assert _is_direct_text(".env") is False
        assert _is_direct_text(".gitignore") is False

    def test_home_directory_path(self):
        """Home directory paths should not be detected as direct text."""
        assert _is_direct_text("~/documents") is False

    def test_single_word_no_extension(self):
        """Single words without extensions default to path (conservative)."""
        assert _is_direct_text("foobar") is False

    def test_existing_path(self, tmp_path):
        """Existing paths should not be detected as direct text."""
        # Create a temporary file
        test_file = tmp_path / "test_existing.txt"
        test_file.write_text("content")

        # Even if the name looks like text, existing paths are not direct text
        assert _is_direct_text(str(test_file)) is False

    def test_path_with_backslash(self):
        """Windows-style paths should not be detected as direct text."""
        assert _is_direct_text("C:\\Users\\docs") is False
        assert _is_direct_text("folder\\file.txt") is False
