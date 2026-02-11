# fitz_ai/engines/fitz_krag/ingestion/strategies/java.py
"""
Java ingestion strategy using tree-sitter.

Extracts classes, interfaces, enums, records, methods, constructors, fields,
and import relationships.
Requires: tree-sitter, tree-sitter-java (optional dependency).
"""

from __future__ import annotations

import logging

import tree_sitter_java as ts_java
from tree_sitter import Language, Parser

from fitz_ai.engines.fitz_krag.ingestion.strategies.base import (
    ImportEdge,
    IngestResult,
    SymbolEntry,
)

logger = logging.getLogger(__name__)

JAVA_LANGUAGE = Language(ts_java.language())


class JavaIngestStrategy:
    """Extracts symbols from Java source using tree-sitter."""

    def content_types(self) -> set[str]:
        return {".java"}

    def extract(self, source: str, file_path: str) -> IngestResult:
        """Extract symbols and imports from Java source."""
        try:
            parser = Parser(JAVA_LANGUAGE)
            tree = parser.parse(source.encode("utf-8"))
        except Exception as e:
            logger.warning(f"Parse error for {file_path}: {e}")
            return IngestResult()

        lines = source.splitlines()
        package = _extract_package(tree.root_node)
        module_name = package or _path_to_module(file_path)
        symbols: list[SymbolEntry] = []
        imports: list[ImportEdge] = []

        _walk_node(tree.root_node, lines, module_name, symbols, imports)
        return IngestResult(symbols=symbols, imports=imports)


def _walk_node(node, lines, module_name, symbols, imports, class_name=None):
    """Recursively walk tree-sitter nodes to extract symbols."""
    for child in node.children:
        node_type = child.type

        if node_type == "class_declaration":
            sym = _extract_class(child, lines, module_name)
            if sym:
                symbols.append(sym)
                body = child.child_by_field_name("body")
                if body:
                    _walk_node(body, lines, module_name, symbols, imports, sym.name)

        elif node_type == "interface_declaration":
            sym = _extract_interface(child, lines, module_name)
            if sym:
                symbols.append(sym)
                body = child.child_by_field_name("body")
                if body:
                    _walk_node(body, lines, module_name, symbols, imports, sym.name)

        elif node_type == "enum_declaration":
            sym = _extract_enum(child, lines, module_name)
            if sym:
                symbols.append(sym)

        elif node_type == "record_declaration":
            sym = _extract_record(child, lines, module_name)
            if sym:
                symbols.append(sym)

        elif node_type == "method_declaration" and class_name:
            sym = _extract_method(child, lines, module_name, class_name)
            if sym:
                symbols.append(sym)

        elif node_type == "constructor_declaration" and class_name:
            sym = _extract_constructor(child, lines, module_name, class_name)
            if sym:
                symbols.append(sym)

        elif node_type == "field_declaration" and class_name:
            extracted = _extract_field(child, lines, module_name, class_name)
            symbols.extend(extracted)

        elif node_type == "import_declaration":
            edge = _extract_import(child)
            if edge:
                imports.append(edge)


def _extract_class(node, lines, module_name):
    """Extract a class declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    source = "\n".join(lines[start - 1 : end])

    superclass = node.child_by_field_name("superclass")
    interfaces = node.child_by_field_name("interfaces")
    sig_parts = [f"class {name}"]
    if superclass:
        sig_parts.append(f"extends {_node_text(superclass)}")
    if interfaces:
        sig_parts.append(f"implements {_node_text(interfaces)}")

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="class",
        start_line=start,
        end_line=end,
        signature=" ".join(sig_parts),
        source=source,
    )


def _extract_interface(node, lines, module_name):
    """Extract an interface declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    source = "\n".join(lines[start - 1 : end])

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="interface",
        start_line=start,
        end_line=end,
        signature=f"interface {name}",
        source=source,
    )


def _extract_enum(node, lines, module_name):
    """Extract an enum declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    source = "\n".join(lines[start - 1 : end])

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="enum",
        start_line=start,
        end_line=end,
        signature=f"enum {name}",
        source=source,
    )


def _extract_record(node, lines, module_name):
    """Extract a record declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    source = "\n".join(lines[start - 1 : end])

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="record",
        start_line=start,
        end_line=end,
        signature=f"record {name}",
        source=source,
    )


def _extract_method(node, lines, module_name, class_name):
    """Extract a method declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    source = "\n".join(lines[start - 1 : end])

    return_type = node.child_by_field_name("type")
    params = node.child_by_field_name("parameters")
    ret_str = f"{_node_text(return_type)} " if return_type else ""
    sig = f"{ret_str}{name}{_node_text(params) if params else '()'}"

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{class_name}.{name}",
        kind="method",
        start_line=start,
        end_line=end,
        signature=sig,
        source=source,
    )


def _extract_constructor(node, lines, module_name, class_name):
    """Extract a constructor declaration."""
    name_node = node.child_by_field_name("name")
    name = _node_text(name_node) if name_node else class_name
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    source = "\n".join(lines[start - 1 : end])

    params = node.child_by_field_name("parameters")
    sig = f"{name}{_node_text(params) if params else '()'}"

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{class_name}.{name}",
        kind="constructor",
        start_line=start,
        end_line=end,
        signature=sig,
        source=source,
    )


def _extract_field(node, lines, module_name, class_name):
    """Extract field declarations (may have multiple declarators)."""
    fields = []
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    source = "\n".join(lines[start - 1 : end])

    for child in node.children:
        if child.type == "variable_declarator":
            name_node = child.child_by_field_name("name")
            if name_node:
                name = _node_text(name_node)
                fields.append(
                    SymbolEntry(
                        name=name,
                        qualified_name=f"{module_name}.{class_name}.{name}",
                        kind="field",
                        start_line=start,
                        end_line=end,
                        signature=source.strip().rstrip(";"),
                        source=source,
                    )
                )
    return fields


def _extract_import(node):
    """Extract import edge from an import declaration."""
    text = _node_text(node).strip().rstrip(";")
    if text.startswith("import "):
        text = text[7:].strip()
        if text.startswith("static "):
            text = text[7:].strip()

        parts = text.rsplit(".", 1)
        if len(parts) == 2:
            module, name = parts
            return ImportEdge(
                target_module=module,
                import_names=[name] if name != "*" else [],
            )
        return ImportEdge(target_module=text, import_names=[])
    return None


def _extract_package(root_node):
    """Extract package name from root node."""
    for child in root_node.children:
        if child.type == "package_declaration":
            text = _node_text(child).strip().rstrip(";")
            if text.startswith("package "):
                return text[8:].strip()
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
    if path.endswith(".java"):
        path = path[:-5]
    return path.replace("/", ".")
