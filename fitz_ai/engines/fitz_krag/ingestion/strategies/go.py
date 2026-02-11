# fitz_ai/engines/fitz_krag/ingestion/strategies/go.py
"""
Go ingestion strategy using tree-sitter.

Extracts functions, methods (with receiver), types (struct, interface),
const/var declarations, and import relationships.
Requires: tree-sitter, tree-sitter-go (optional dependency).
"""

from __future__ import annotations

import logging

import tree_sitter_go as ts_go
from tree_sitter import Language, Parser

from fitz_ai.engines.fitz_krag.ingestion.strategies.base import (
    ImportEdge,
    IngestResult,
    SymbolEntry,
)

logger = logging.getLogger(__name__)

GO_LANGUAGE = Language(ts_go.language())


class GoIngestStrategy:
    """Extracts symbols from Go source using tree-sitter."""

    def content_types(self) -> set[str]:
        return {".go"}

    def extract(self, source: str, file_path: str) -> IngestResult:
        """Extract symbols and imports from Go source."""
        try:
            parser = Parser(GO_LANGUAGE)
            tree = parser.parse(source.encode("utf-8"))
        except Exception as e:
            logger.warning(f"Parse error for {file_path}: {e}")
            return IngestResult()

        lines = source.splitlines()
        package = _extract_package(tree.root_node)
        module_name = package or _path_to_module(file_path)
        symbols: list[SymbolEntry] = []
        imports: list[ImportEdge] = []

        for child in tree.root_node.children:
            node_type = child.type

            if node_type == "function_declaration":
                sym = _extract_function(child, lines, module_name)
                if sym:
                    symbols.append(sym)

            elif node_type == "method_declaration":
                sym = _extract_method(child, lines, module_name)
                if sym:
                    symbols.append(sym)

            elif node_type == "type_declaration":
                extracted = _extract_type_decl(child, lines, module_name)
                symbols.extend(extracted)

            elif node_type == "const_declaration":
                extracted = _extract_const_var(child, lines, module_name, "constant")
                symbols.extend(extracted)

            elif node_type == "var_declaration":
                extracted = _extract_const_var(child, lines, module_name, "variable")
                symbols.extend(extracted)

            elif node_type == "import_declaration":
                edges = _extract_imports(child)
                imports.extend(edges)

        return IngestResult(symbols=symbols, imports=imports)


def _extract_function(node, lines, module_name):
    """Extract a top-level function declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    source = "\n".join(lines[start - 1 : end])

    params = node.child_by_field_name("parameters")
    result = node.child_by_field_name("result")
    sig = f"func {name}{_node_text(params) if params else '()'}"
    if result:
        sig += f" {_node_text(result)}"

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="function",
        start_line=start,
        end_line=end,
        signature=sig,
        source=source,
    )


def _extract_method(node, lines, module_name):
    """Extract a method declaration (function with receiver)."""
    name_node = node.child_by_field_name("name")
    receiver_node = node.child_by_field_name("receiver")
    if not name_node:
        return None

    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    source = "\n".join(lines[start - 1 : end])

    receiver_type = _extract_receiver_type(receiver_node)
    params = node.child_by_field_name("parameters")
    result = node.child_by_field_name("result")

    receiver_str = f"({_node_text(receiver_node)})" if receiver_node else "()"
    sig = f"func {receiver_str} {name}{_node_text(params) if params else '()'}"
    if result:
        sig += f" {_node_text(result)}"

    qualified = (
        f"{module_name}.{receiver_type}.{name}" if receiver_type else f"{module_name}.{name}"
    )

    return SymbolEntry(
        name=name,
        qualified_name=qualified,
        kind="method",
        start_line=start,
        end_line=end,
        signature=sig,
        source=source,
    )


def _extract_type_decl(node, lines, module_name):
    """Extract type declarations (struct, interface, etc.)."""
    symbols = []
    for child in node.children:
        if child.type == "type_spec":
            name_node = child.child_by_field_name("name")
            type_node = child.child_by_field_name("type")
            if not name_node:
                continue

            name = _node_text(name_node)
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            source = "\n".join(lines[start - 1 : end])

            if type_node and type_node.type == "struct_type":
                kind = "struct"
                sig = f"type {name} struct"
            elif type_node and type_node.type == "interface_type":
                kind = "interface"
                sig = f"type {name} interface"
            else:
                kind = "type"
                sig = f"type {name}"

            symbols.append(
                SymbolEntry(
                    name=name,
                    qualified_name=f"{module_name}.{name}",
                    kind=kind,
                    start_line=start,
                    end_line=end,
                    signature=sig,
                    source=source,
                )
            )
    return symbols


def _extract_const_var(node, lines, module_name, kind):
    """Extract const or var declarations."""
    symbols = []
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    source = "\n".join(lines[start - 1 : end])

    for child in node.children:
        if child.type in ("const_spec", "var_spec"):
            name_node = child.child_by_field_name("name")
            if name_node:
                name = _node_text(name_node)
                symbols.append(
                    SymbolEntry(
                        name=name,
                        qualified_name=f"{module_name}.{name}",
                        kind=kind,
                        start_line=start,
                        end_line=end,
                        signature=f"{kind} {name}",
                        source=source,
                    )
                )
    return symbols


def _extract_imports(node):
    """Extract import edges from an import declaration."""
    edges = []
    for child in node.children:
        if child.type == "import_spec":
            path_node = child.child_by_field_name("path")
            if path_node:
                path = _node_text(path_node).strip('"')
                name = path.rsplit("/", 1)[-1]
                edges.append(ImportEdge(target_module=path, import_names=[name]))
        elif child.type == "import_spec_list":
            for spec in child.children:
                if spec.type == "import_spec":
                    path_node = spec.child_by_field_name("path")
                    if path_node:
                        path = _node_text(path_node).strip('"')
                        name = path.rsplit("/", 1)[-1]
                        edges.append(ImportEdge(target_module=path, import_names=[name]))
    return edges


def _extract_package(root_node):
    """Extract package name from root node."""
    for child in root_node.children:
        if child.type == "package_clause":
            for sub in child.children:
                if sub.type == "package_identifier":
                    return _node_text(sub)
    return None


def _extract_receiver_type(receiver_node):
    """Extract the type name from a method receiver."""
    if not receiver_node:
        return None
    text = _node_text(receiver_node)
    text = text.strip("()")
    parts = text.split()
    for part in reversed(parts):
        clean = part.lstrip("*")
        if clean and clean[0].isupper():
            return clean
    return None


def _node_text(node) -> str:
    """Get text content of a tree-sitter node."""
    if node is None:
        return ""
    if isinstance(node.text, bytes):
        return node.text.decode("utf-8")
    return str(node.text)


def _path_to_module(file_path: str) -> str:
    """Convert file path to package-like name."""
    path = file_path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    if path.endswith(".go"):
        path = path[:-3]
    return path.replace("/", ".")
