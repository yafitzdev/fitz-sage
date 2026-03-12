# tests/unit/test_context_compressor.py
"""Tests for AST-based code compression."""

from fitz_ai.engines.fitz_krag.context.compressor import (
    compress_python,
    compress_results,
    _strip_comments_and_blanks,
)
from fitz_ai.engines.fitz_krag.types import Address, AddressKind, ReadResult


def _make_result(content: str, kind=AddressKind.SYMBOL, path="foo.py") -> ReadResult:
    return ReadResult(
        address=Address(
            kind=kind, source_id="f1", location="test", summary="test", score=0.8
        ),
        content=content,
        file_path=path,
    )


class TestCompressPython:
    def test_strips_docstrings(self):
        source = '''def foo():
    """This is a docstring."""
    return 1
'''
        result = compress_python(source)
        assert '"""' not in result
        assert "def foo():" in result
        assert "return 1" in result

    def test_collapses_long_bodies(self):
        lines = ["    x = {}\n".format(i) for i in range(20)]
        source = "def foo():\n" + "".join(lines)
        result = compress_python(source)
        assert "def foo():" in result
        assert "...  # " in result
        assert "x = 0" not in result

    def test_keeps_short_bodies(self):
        source = "def foo():\n    x = 1\n    return x\n"
        result = compress_python(source)
        assert "x = 1" in result
        assert "return x" in result

    def test_keeps_imports(self):
        source = "import os\nfrom pathlib import Path\n\ndef foo():\n    pass\n"
        result = compress_python(source)
        assert "import os" in result
        assert "from pathlib import Path" in result

    def test_keeps_class_signatures(self):
        source = '''class Foo:
    """Docstring."""

    def __init__(self):
        self.x = 1
'''
        result = compress_python(source)
        assert "class Foo:" in result
        assert "def __init__(self):" in result
        assert '"""Docstring."""' not in result

    def test_handles_syntax_errors(self):
        source = "this is not valid python {{{"
        result = compress_python(source)
        assert result  # Returns something (stripped version)

    def test_strips_comments(self):
        source = "# this is a comment\nx = 1\n# another comment\ny = 2\n"
        result = compress_python(source)
        assert "# this is a comment" not in result
        assert "x = 1" in result
        assert "y = 2" in result

    def test_collapses_blank_lines(self):
        source = "x = 1\n\n\n\n\ny = 2\n"
        result = compress_python(source)
        lines = result.splitlines()
        # Should have at most one blank line between x and y
        blank_count = sum(1 for ln in lines if not ln.strip())
        assert blank_count <= 1

    def test_keeps_decorators(self):
        source = "@staticmethod\ndef foo():\n    return 1\n"
        result = compress_python(source)
        assert "@staticmethod" in result
        assert "def foo():" in result

    def test_keeps_constants(self):
        source = 'MAX_SIZE = 100\nDEFAULT_NAME = "test"\n'
        result = compress_python(source)
        assert "MAX_SIZE = 100" in result
        assert 'DEFAULT_NAME = "test"' in result


class TestStripCommentsAndBlanks:
    def test_removes_comments(self):
        result = _strip_comments_and_blanks("# comment\nx = 1\n")
        assert "# comment" not in result
        assert "x = 1" in result

    def test_keeps_shebangs(self):
        result = _strip_comments_and_blanks("#!/usr/bin/env python\nx = 1\n")
        assert "#!/usr/bin/env python" in result

    def test_keeps_type_ignore(self):
        result = _strip_comments_and_blanks("# type: ignore\nx = 1\n")
        assert "# type: ignore" in result


class TestCompressResults:
    def test_compresses_python_code(self):
        source = '''def foo():
    """Long docstring."""
    x = 1
    return x
'''
        result = _make_result(source)
        compressed = compress_results([result])
        assert len(compressed) == 1
        assert '"""' not in compressed[0].content

    def test_skips_non_python(self):
        result = _make_result("some markdown content", path="readme.md")
        compressed = compress_results([result])
        assert compressed[0].content == "some markdown content"

    def test_skips_sections(self):
        result = _make_result("section text", kind=AddressKind.SECTION, path="doc.pdf")
        compressed = compress_results([result])
        assert compressed[0].content == "section text"

    def test_preserves_metadata(self):
        result = _make_result("x = 1\n")
        result.metadata["score"] = 0.9
        compressed = compress_results([result])
        assert compressed[0].metadata["score"] == 0.9
        assert compressed[0].file_path == "foo.py"
        assert compressed[0].address.source_id == "f1"
