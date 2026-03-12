# tests/unit/test_code_retriever.py
"""Tests for standalone CodeRetriever (fitz-ai[code])."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fitz_ai.code.indexer import (
    build_file_list,
    build_import_graph,
    build_structural_index,
)
from fitz_ai.code.retriever import CodeRetriever
from fitz_ai.engines.fitz_krag.types import AddressKind


# ---------------------------------------------------------------------------
# Indexer tests
# ---------------------------------------------------------------------------


class TestBuildFileList:
    def test_finds_python_files(self, tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text("x = 1")
        (tmp_path / "app" / "utils.py").write_text("y = 2")

        files = build_file_list(tmp_path)
        assert "app/main.py" in files
        assert "app/utils.py" in files

    def test_skips_hidden_dirs(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config.py").write_text("x = 1")
        (tmp_path / "app.py").write_text("y = 2")

        files = build_file_list(tmp_path)
        assert "app.py" in files
        assert not any(".git" in f for f in files)

    def test_skips_pycache(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.cpython-312.pyc").write_text("")
        (tmp_path / "mod.py").write_text("x = 1")

        files = build_file_list(tmp_path)
        assert "mod.py" in files
        assert not any("__pycache__" in f for f in files)

    def test_respects_max_files(self, tmp_path):
        for i in range(10):
            (tmp_path / f"mod{i}.py").write_text(f"x = {i}")

        files = build_file_list(tmp_path, max_files=3)
        assert len(files) == 3


class TestBuildStructuralIndex:
    def test_extracts_python_classes(self, tmp_path):
        (tmp_path / "models.py").write_text(textwrap.dedent("""\
            class User:
                def __init__(self, name: str) -> None:
                    self.name = name

                def greet(self) -> str:
                    return f"Hello {self.name}"
        """))

        index = build_structural_index(tmp_path, ["models.py"])
        assert "## models.py" in index
        assert "User" in index
        assert "classes:" in index

    def test_extracts_python_functions(self, tmp_path):
        (tmp_path / "utils.py").write_text(textwrap.dedent("""\
            def slugify(text: str) -> str:
                return text.lower().replace(" ", "-")
        """))

        index = build_structural_index(tmp_path, ["utils.py"])
        assert "slugify" in index
        assert "functions:" in index

    def test_extracts_imports(self, tmp_path):
        (tmp_path / "main.py").write_text("import os\nfrom pathlib import Path\n")

        index = build_structural_index(tmp_path, ["main.py"])
        assert "imports:" in index
        assert "os" in index

    def test_handles_syntax_errors(self, tmp_path):
        (tmp_path / "bad.py").write_text("class {{ invalid python")

        index = build_structural_index(tmp_path, ["bad.py"])
        assert "## bad.py" in index  # File still appears

    def test_truncation_under_budget(self, tmp_path):
        (tmp_path / "a.py").write_text("class A:\n    pass\n")
        (tmp_path / "b.py").write_text("class B:\n    pass\n")

        index = build_structural_index(tmp_path, ["a.py", "b.py"], max_chars=50)
        assert "## a.py" in index
        assert "## b.py" in index


class TestBuildImportGraph:
    def test_resolves_intra_project_imports(self, tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "__init__.py").write_text("")
        (tmp_path / "app" / "models.py").write_text("class User: pass\n")
        (tmp_path / "app" / "views.py").write_text("from app.models import User\n")

        files = ["app/__init__.py", "app/models.py", "app/views.py"]
        graph = build_import_graph(tmp_path, files)

        assert "app/views.py" in graph
        assert "app/models.py" in graph["app/views.py"]

    def test_ignores_external_imports(self, tmp_path):
        (tmp_path / "main.py").write_text("import numpy\nimport os\n")

        graph = build_import_graph(tmp_path, ["main.py"])
        # numpy and os are external — not in the graph
        assert "main.py" not in graph

    def test_forward_only(self, tmp_path):
        (tmp_path / "a.py").write_text("from b import x\n")
        (tmp_path / "b.py").write_text("x = 1\n")

        graph = build_import_graph(tmp_path, ["a.py", "b.py"])
        assert "b.py" in graph.get("a.py", set())
        # b.py doesn't import a.py
        assert "a.py" not in graph.get("b.py", set())


# ---------------------------------------------------------------------------
# CodeRetriever tests
# ---------------------------------------------------------------------------


def _make_chat_factory(response=None):
    if response is None:
        response = json.dumps({"search_terms": ["test"], "files": []})
    chat = MagicMock()
    chat.chat.return_value = response
    factory = MagicMock(return_value=chat)
    return factory


class TestCodeRetriever:
    def test_retrieve_end_to_end(self, tmp_path):
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "main.py").write_text(textwrap.dedent("""\
            from app.utils import helper

            def run():
                return helper()
        """))
        (tmp_path / "app" / "utils.py").write_text(textwrap.dedent("""\
            def helper():
                return "hello"
        """))
        (tmp_path / "app" / "__init__.py").write_text("")

        response = json.dumps({
            "search_terms": ["run", "helper"],
            "files": ["app/main.py"],
        })
        retriever = CodeRetriever(
            source_dir=tmp_path,
            chat_factory=_make_chat_factory(response),
        )
        results = retriever.retrieve("How does run() work?")

        assert len(results) >= 1
        paths = [r.file_path for r in results]
        assert "app/main.py" in paths
        # Import expansion should pull in utils.py
        assert "app/utils.py" in paths

    def test_returns_file_addresses(self, tmp_path):
        (tmp_path / "main.py").write_text("def foo(): pass\n")
        response = json.dumps({"search_terms": [], "files": ["main.py"]})

        retriever = CodeRetriever(tmp_path, _make_chat_factory(response))
        results = retriever.retrieve("What does foo do?")

        assert len(results) == 1
        assert results[0].address.kind == AddressKind.FILE

    def test_compresses_python(self, tmp_path):
        source = textwrap.dedent('''\
            def foo():
                """Long docstring that should be stripped."""
                x = 1
                return x
        ''')
        (tmp_path / "mod.py").write_text(source)
        response = json.dumps({"search_terms": [], "files": ["mod.py"]})

        retriever = CodeRetriever(tmp_path, _make_chat_factory(response))
        results = retriever.retrieve("What does foo do?")

        assert len(results) == 1
        assert '"""' not in results[0].content  # Docstring stripped
        assert "def foo():" in results[0].content

    def test_origin_scores(self, tmp_path):
        (tmp_path / "a.py").write_text("from b import x\n")
        (tmp_path / "b.py").write_text("x = 1\n")

        response = json.dumps({"search_terms": [], "files": ["a.py"]})
        retriever = CodeRetriever(tmp_path, _make_chat_factory(response))
        results = retriever.retrieve("What is x?")

        origins = {r.file_path: r.address.metadata["origin"] for r in results}
        assert origins.get("a.py") == "selected"
        assert origins.get("b.py") == "import"
        assert results[0].address.score == 1.0  # selected

    def test_empty_llm_response_returns_empty(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1\n")
        response = json.dumps({"search_terms": [], "files": []})

        retriever = CodeRetriever(tmp_path, _make_chat_factory(response))
        results = retriever.retrieve("anything")

        assert results == []

    def test_structural_index_accessible(self, tmp_path):
        (tmp_path / "mod.py").write_text("class Foo: pass\n")

        retriever = CodeRetriever(tmp_path, _make_chat_factory())
        index = retriever.get_structural_index()

        assert "## mod.py" in index
        assert "Foo" in index

    def test_neighbor_expansion(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "a.py").write_text("class A: pass\n")
        (tmp_path / "pkg" / "b.py").write_text("class B: pass\n")
        (tmp_path / "pkg" / "c.py").write_text("class C: pass\n")

        response = json.dumps({"search_terms": [], "files": ["pkg/a.py"]})
        retriever = CodeRetriever(tmp_path, _make_chat_factory(response))
        results = retriever.retrieve("What is A?")

        paths = {r.file_path for r in results}
        # b.py and c.py are neighbors of a.py
        assert "pkg/b.py" in paths
        assert "pkg/c.py" in paths


class TestNoHeavyImports:
    def test_code_module_does_not_import_heavy_deps(self):
        """Verify fitz_ai.code doesn't transitively import psycopg/pgvector/docling."""
        heavy = ["psycopg", "pgvector", "docling", "fitz_pgserver"]
        loaded = [m for m in heavy if any(k.startswith(m) for k in sys.modules)]
        assert not loaded, f"Heavy modules loaded: {loaded}"
