# fitz_ai/code/indexer.py
"""
File-based structural index and import graph builder.

Extracts structural information from source files using AST (Python) and
regex patterns (other languages). No database dependencies.

Adapted from fitz-graveyard's indexer.py for standalone use in fitz-ai.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

_MAX_INDEX_CHARS = 120_000

_PYTHON_EXTS = {".py"}
_CONFIG_EXTS = {".yaml", ".yml", ".json", ".toml"}
_MARKDOWN_EXTS = {".md", ".rst"}
_GENERIC_CODE_EXTS = {
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rs",
    ".java", ".kt", ".scala",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    ".rb", ".cs", ".swift", ".php",
    ".lua", ".zig",
    ".sh", ".bash", ".zsh",
}

INDEXABLE_EXTENSIONS = _PYTHON_EXTS | _CONFIG_EXTS | _MARKDOWN_EXTS | _GENERIC_CODE_EXTS

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".env",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".eggs", "*.egg-info",
}

_KEY_DECORATORS = frozenset({
    "dataclass", "abstractmethod", "property", "staticmethod",
    "classmethod", "override",
})


def build_file_list(source_dir: Path, max_files: int = 2000) -> list[str]:
    """Walk source_dir and return relative posix paths of indexable files."""
    root = source_dir.resolve()
    files: list[str] = []

    for path in sorted(root.rglob("*")):
        if len(files) >= max_files:
            break
        if not path.is_file():
            continue
        # Skip hidden/ignored directories
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if rel.suffix.lower() in INDEXABLE_EXTENSIONS:
            files.append(rel.as_posix())

    return files


def build_structural_index(
    source_dir: Path,
    file_paths: list[str],
    max_file_bytes: int = 50_000,
    max_chars: int = _MAX_INDEX_CHARS,
    connection_counts: dict[str, int] | None = None,
) -> str:
    """Build a compact structural index from files on disk.

    Args:
        source_dir: Root directory of the codebase.
        file_paths: Relative posix paths to index.
        max_file_bytes: Max bytes to read per file.
        max_chars: Max total index size in characters.
        connection_counts: Optional import connection counts for truncation priority.

    Returns:
        Multi-line text index with structural info per file.
    """
    root = source_dir.resolve()
    entries: list[tuple[str, str]] = []

    for rel_path in file_paths:
        full_path = root / rel_path
        if not full_path.is_file():
            continue
        try:
            raw = full_path.read_bytes()[:max_file_bytes]
            content = raw.decode("utf-8", errors="replace")
        except OSError:
            continue
        if not content.strip():
            continue

        suffix = PurePosixPath(rel_path).suffix.lower()
        info = _extract_structure(suffix, content)
        entries.append((rel_path, info or "(no structural info)"))

    return _format_index(entries, max_chars, connection_counts)


def build_import_graph(
    source_dir: Path,
    file_paths: list[str],
    max_file_bytes: int = 50_000,
) -> dict[str, set[str]]:
    """Build forward import map from Python files.

    Returns:
        {file_path: {imported_file_path, ...}} — intra-project imports only.
    """
    root = source_dir.resolve()
    module_lookup = _build_module_file_lookup(file_paths)
    forward: dict[str, set[str]] = {}

    for rel_path in file_paths:
        if not rel_path.endswith(".py"):
            continue
        full_path = root / rel_path
        if not full_path.is_file():
            continue
        try:
            raw = full_path.read_bytes()[:max_file_bytes]
            content = raw.decode("utf-8", errors="replace")
        except OSError:
            continue

        full_imports = _extract_full_imports(content)
        resolved = set()
        for imp in full_imports:
            target = module_lookup.get(imp)
            if target and target != rel_path:
                resolved.add(target)

        if resolved:
            forward[rel_path] = resolved

    return forward


# ---------------------------------------------------------------------------
# Python AST extraction
# ---------------------------------------------------------------------------


def _extract_structure(suffix: str, content: str) -> str:
    if suffix in _PYTHON_EXTS:
        return _extract_python(content)
    if suffix in _CONFIG_EXTS:
        return _extract_config(suffix, content)
    if suffix in _MARKDOWN_EXTS:
        return _extract_markdown(content)
    if suffix in _GENERIC_CODE_EXTS:
        return _extract_generic_code(content)
    return ""


def _extract_python(content: str) -> str:
    """Extract structure from Python files using AST."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _extract_python_regex(content)

    lines: list[str] = []

    module_doc = ast.get_docstring(tree)
    if module_doc:
        first_line = module_doc.strip().splitlines()[0]
        lines.append(f'doc: "{first_line}"')

    classes = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = [_ast_name(b) for b in node.bases]
            methods = []
            for n in ast.iter_child_nodes(node):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_str = n.name
                    if n.returns:
                        try:
                            m_str += f" -> {ast.unparse(n.returns)}"
                        except Exception:
                            m_str += f" -> {_ast_name(n.returns)}"
                    methods.append(m_str)
            cls_str = node.name
            if bases:
                cls_str += f"({', '.join(bases)})"
            decs = _extract_key_decorators(node.decorator_list)
            if decs:
                cls_str += f" [{', '.join(f'@{d}' for d in decs)}]"
            if methods:
                cls_str += f" [{', '.join(methods)}]"
            classes.append(cls_str)
    if classes:
        lines.append(f"classes: {'; '.join(classes)}")

    functions = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [arg.arg for arg in node.args.args if arg.arg != "self"]
            func_str = f"{node.name}({', '.join(params)})"
            if node.returns:
                try:
                    func_str += f" -> {ast.unparse(node.returns)}"
                except Exception:
                    func_str += f" -> {_ast_name(node.returns)}"
            decs = _extract_key_decorators(node.decorator_list)
            if decs:
                func_str += f" [{', '.join(f'@{d}' for d in decs)}]"
            functions.append(func_str)
    if functions:
        lines.append(f"functions: {', '.join(functions)}")

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    if imports:
        lines.append(f"imports: {', '.join(sorted(imports))}")

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        names = [
                            elt.value for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
                        if names:
                            lines.append(f"exports: {', '.join(names)}")

    return "\n".join(lines)


def _ast_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _ast_name(node.value)
        return f"{value}.{node.attr}" if value else node.attr
    if isinstance(node, ast.Subscript):
        return _ast_name(node.value)
    return "?"


def _extract_key_decorators(decorator_list: list[ast.expr]) -> list[str]:
    result = []
    for dec in decorator_list:
        name = None
        if isinstance(dec, ast.Name):
            name = dec.id
        elif isinstance(dec, ast.Attribute):
            name = dec.attr
        elif isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
        if name and name in _KEY_DECORATORS:
            result.append(name)
    return result


def _extract_python_regex(content: str) -> str:
    lines: list[str] = []
    classes = re.findall(r'^class\s+(\w+)(?:\(([^)]*)\))?:', content, re.MULTILINE)
    if classes:
        cls_strs = [f"{n}({b})" if b else n for n, b in classes]
        lines.append(f"classes: {'; '.join(cls_strs)}")
    functions = re.findall(r'^(?:async\s+)?def\s+(\w+)\(([^)]*)\)', content, re.MULTILINE)
    if functions:
        lines.append(f"functions: {', '.join(f'{n}({p})' for n, p in functions)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Non-Python extractors
# ---------------------------------------------------------------------------


def _extract_config(suffix: str, content: str) -> str:
    try:
        if suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(content)
        elif suffix == ".json":
            data = json.loads(content)
        elif suffix == ".toml":
            import tomllib
            data = tomllib.loads(content)
        else:
            return ""
    except Exception:
        return ""
    if isinstance(data, dict):
        keys = list(data.keys())[:20]
        return f"keys: {', '.join(str(k) for k in keys)}"
    return ""


def _extract_markdown(content: str) -> str:
    headings = re.findall(r'^(#{1,3})\s+(.+)', content, re.MULTILINE)
    if headings:
        items = [f"{'#' * len(h[0])} {h[1].strip()}" for h in headings[:15]]
        return f"headings: {'; '.join(items)}"
    rst_headings = re.findall(r'^(.+)\n[=\-~^]+$', content, re.MULTILINE)
    if rst_headings:
        return f"headings: {'; '.join(h.strip() for h in rst_headings[:15])}"
    return ""


def _extract_generic_code(content: str) -> str:
    lines: list[str] = []
    type_defs = re.findall(
        r'^(?:export\s+)?(?:pub\s+)?(?:public\s+|private\s+|protected\s+|abstract\s+)?'
        r'(?:class|struct|interface|trait|enum|type)\s+(\w+)',
        content, re.MULTILINE,
    )
    if type_defs:
        lines.append(f"types: {', '.join(dict.fromkeys(type_defs))}")

    func_patterns = [
        r'^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(',
        r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*[<(]',
        r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[<(]',
        r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|function)',
        r'^\s+(?:public|private|protected|static|override|virtual|abstract|final|\s)*'
        r'(?:fun|void|int|string|bool|float|double|var|val|Task|async)\s+(\w+)\s*[<(]',
        r'^(?:\s+)?def\s+(\w+)',
        r'^(?:static\s+)?(?:inline\s+)?(?:const\s+)?\w[\w:*&<> ]*\s+(\w+)\s*\([^;]*$',
    ]
    skip = {"if", "for", "while", "switch", "return", "main"}
    functions: list[str] = []
    for pat in func_patterns:
        for m in re.finditer(pat, content, re.MULTILINE):
            name = m.group(1)
            if name not in functions and name not in skip:
                functions.append(name)
    if functions:
        lines.append(f"functions: {', '.join(functions[:20])}")

    imports: set[str] = set()
    import_patterns = [
        r'^import\s+["\']([^"\']+)["\']',
        r'^import\s+.*\s+from\s+["\']([^"\']+)["\']',
        r'^(?:const|let|var)\s+.*=\s*require\(["\']([^"\']+)["\']\)',
        r'^import\s+"([^"]+)"',
        r'^use\s+([\w:]+)',
        r'^import\s+([\w.]+)',
        r'^using\s+([\w.]+)',
        r'^require\s+["\']([^"\']+)["\']',
        r'^#include\s+[<"]([^>"]+)[>"]',
    ]
    for pat in import_patterns:
        for m in re.finditer(pat, content, re.MULTILINE):
            mod = m.group(1).split("/")[0].split("::")[0].split(".")[0]
            if mod:
                imports.add(mod)
    if imports:
        lines.append(f"imports: {', '.join(sorted(imports))}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Import graph helpers
# ---------------------------------------------------------------------------


def _extract_full_imports(content: str) -> set[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _extract_full_imports_regex(content)

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _extract_full_imports_regex(content: str) -> set[str]:
    imports: set[str] = set()
    for m in re.finditer(r'^\s*from\s+(\S+)\s+import', content, re.MULTILINE):
        imports.add(m.group(1))
    for m in re.finditer(r'^\s*import\s+(\S+)', content, re.MULTILINE):
        imports.add(m.group(1).split(",")[0].strip())
    return imports


def _build_module_file_lookup(file_list: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for rel_path in file_list:
        if not rel_path.endswith(".py"):
            continue
        module = rel_path[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            lookup[module] = rel_path
            lookup[module[:-9]] = rel_path
        else:
            lookup[module] = rel_path
    return lookup


# ---------------------------------------------------------------------------
# Index formatting with truncation
# ---------------------------------------------------------------------------


def _format_index(
    entries: list[tuple[str, str]],
    max_chars: int = _MAX_INDEX_CHARS,
    connection_counts: dict[str, int] | None = None,
) -> str:
    parts = [f"## {rel_path}\n{info}" for rel_path, info in entries]
    full = "\n\n".join(parts)

    if len(full) <= max_chars:
        return full

    mutable = list(entries)
    conns = connection_counts or {}
    by_priority = sorted(
        range(len(mutable)),
        key=lambda i: conns.get(mutable[i][0], 0),
    )

    for idx in by_priority:
        rel_path, info = mutable[idx]
        lines = [ln for ln in info.splitlines() if not ln.startswith("imports:")]
        mutable[idx] = (rel_path, "\n".join(lines))
        if _estimate_size(mutable) <= max_chars:
            break

    if _estimate_size(mutable) > max_chars:
        for idx in by_priority:
            rel_path, info = mutable[idx]
            lines = [ln for ln in info.splitlines() if not ln.startswith("functions:")]
            mutable[idx] = (rel_path, "\n".join(lines))
            if _estimate_size(mutable) <= max_chars:
                break

    if _estimate_size(mutable) > max_chars:
        for idx in by_priority:
            mutable[idx] = (mutable[idx][0], "")
            if _estimate_size(mutable) <= max_chars:
                break

    result_parts = []
    for rel_path, info in mutable:
        if info.strip():
            result_parts.append(f"## {rel_path}\n{info}")
        else:
            result_parts.append(f"## {rel_path}")
    return "\n\n".join(result_parts)


def _estimate_size(entries: list[tuple[str, str]]) -> int:
    return sum(3 + len(p) + 1 + len(i) + 2 for p, i in entries)
