# fitz_ai/plugin_gen/library_context.py
"""
Library context fetcher for plugin generation.

Detects library mentions in user queries and fetches documentation
from PyPI/GitHub to provide context for generating plugins that
use external dependencies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Known libraries with their canonical PyPI names
KNOWN_LIBRARIES = {
    # Document parsing
    "docling": "docling",
    "pymupdf": "PyMuPDF",
    "fitz": "PyMuPDF",
    "unstructured": "unstructured",
    "marker": "marker-pdf",
    "marker-pdf": "marker-pdf",
    "pdfplumber": "pdfplumber",
    "pypdf": "pypdf",
    "pdf2image": "pdf2image",
    # OCR
    "tesseract": "pytesseract",
    "pytesseract": "pytesseract",
    "easyocr": "easyocr",
    "paddleocr": "paddleocr",
    # NLP / Embeddings
    "sentence-transformers": "sentence-transformers",
    "transformers": "transformers",
    "spacy": "spacy",
    "nltk": "nltk",
    # Vector DBs
    "qdrant": "qdrant-client",
    "pinecone": "pinecone-client",
    "weaviate": "weaviate-client",
    "chromadb": "chromadb",
    "milvus": "pymilvus",
    # LLM clients
    "openai": "openai",
    "anthropic": "anthropic",
    "cohere": "cohere",
    "ollama": "ollama",
    # Other
    "langchain": "langchain",
    "llamaindex": "llama-index",
    "llama-index": "llama-index",
}


@dataclass
class LibraryContext:
    """Context about an external library for plugin generation."""

    name: str
    pypi_name: str
    install_command: str
    summary: str
    readme_excerpt: str

    def __str__(self) -> str:
        return f"LibraryContext({self.name})"


def detect_library_mentions(query: str) -> List[str]:
    """
    Detect explicit library mentions in a user query.

    Only returns libraries that are explicitly named, not inferred.

    Args:
        query: User's plugin generation query

    Returns:
        List of detected library names (normalized to lowercase)

    Examples:
        >>> detect_library_mentions("pdf chunker using docling")
        ['docling']
        >>> detect_library_mentions("sentence chunker")
        []
        >>> detect_library_mentions("use pymupdf and pytesseract")
        ['pymupdf', 'pytesseract']
    """
    query_lower = query.lower()
    detected = []

    # Patterns that indicate explicit library usage
    explicit_patterns = [
        r"using\s+(?:the\s+)?(\w+(?:-\w+)?)",
        r"with\s+(?:the\s+)?(\w+(?:-\w+)?)\s+(?:library|package|module)",
        r"use\s+(?:the\s+)?(\w+(?:-\w+)?)",
        r"via\s+(?:the\s+)?(\w+(?:-\w+)?)",
        r"(\w+(?:-\w+)?)\s+library",
        r"(\w+(?:-\w+)?)\s+package",
    ]

    # Extract candidates from patterns
    candidates = set()
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, query_lower):
            candidates.add(match.group(1))

    # Also check for direct library name mentions
    for lib_name in KNOWN_LIBRARIES:
        if lib_name in query_lower:
            candidates.add(lib_name)

    # Filter to only known libraries
    for candidate in candidates:
        if candidate in KNOWN_LIBRARIES:
            if candidate not in detected:
                detected.append(candidate)

    logger.debug(f"Detected libraries in query: {detected}")
    return detected


def fetch_library_context(name: str) -> Optional[LibraryContext]:
    """
    Fetch library documentation from PyPI.

    Args:
        name: Library name (as detected, e.g., "docling")

    Returns:
        LibraryContext with documentation, or None if fetch fails
    """
    pypi_name = KNOWN_LIBRARIES.get(name, name)

    try:
        import httpx
    except ImportError:
        logger.warning("httpx not available, cannot fetch library context")
        return None

    try:
        # Fetch from PyPI JSON API
        url = f"https://pypi.org/pypi/{pypi_name}/json"
        logger.info(f"Fetching library context from {url}")

        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)

            if response.status_code != 200:
                logger.warning(f"PyPI returned {response.status_code} for {pypi_name}")
                return None

            data = response.json()

        info = data.get("info", {})
        summary = info.get("summary", "")
        description = info.get("description", "")

        # Extract a useful excerpt from the description
        readme_excerpt = _extract_readme_excerpt(description, max_chars=3000)

        return LibraryContext(
            name=name,
            pypi_name=pypi_name,
            install_command=f"pip install {pypi_name}",
            summary=summary,
            readme_excerpt=readme_excerpt,
        )

    except Exception as e:
        logger.warning(f"Failed to fetch library context for {name}: {e}")
        return None


def _extract_readme_excerpt(description: str, max_chars: int = 4000) -> str:
    """
    Extract the most useful parts of a README for plugin generation.

    Prioritizes: code examples, API usage, import statements.
    Strips: badges, links, prose, changelogs, HTML.
    """
    if not description:
        return ""

    # Strip HTML tags first
    description = re.sub(r"<[^>]+>", "", description)

    # Extract code blocks first - these are most valuable
    code_blocks = _extract_code_blocks(description)

    # Build output prioritizing code
    parts = []
    chars_used = 0

    # Add code examples first (most important)
    if code_blocks:
        parts.append("## API Usage Examples\n")
        chars_used += 25

        for lang, code in code_blocks:
            block = f"```{lang}\n{code}\n```\n"
            if chars_used + len(block) <= max_chars * 0.7:  # Reserve 30% for context
                parts.append(block)
                chars_used += len(block)

    # Add brief context from useful sections
    context = _extract_context_sections(description, max_chars - chars_used)
    if context:
        parts.append("\n## Additional Context\n")
        parts.append(context)

    return "\n".join(parts).strip()


def _extract_code_blocks(text: str) -> list:
    """
    Extract all code blocks from markdown text.

    Returns list of (language, code) tuples, prioritizing Python.
    """
    # Pattern for fenced code blocks: ```lang\ncode\n```
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    code_blocks = []
    for lang, code in matches:
        lang = lang.lower() or "python"
        code = code.strip()

        # Skip empty or very short blocks
        if len(code) < 10:
            continue

        # Skip shell-only blocks (just pip install, etc.)
        if lang in ("bash", "sh", "shell", "console"):
            if code.startswith("pip ") or code.startswith("$ pip"):
                continue

        code_blocks.append((lang, code))

    # Sort: Python first, then by length (longer = more complete examples)
    def sort_key(item):
        lang, code = item
        lang_priority = 0 if lang == "python" else 1
        # Prefer blocks with imports and function calls
        has_import = 1 if "import " in code else 0
        has_call = 1 if "(" in code and ")" in code else 0
        return (lang_priority, -has_import, -has_call, -len(code))

    code_blocks.sort(key=sort_key)

    return code_blocks


def _extract_context_sections(text: str, max_chars: int) -> str:
    """
    Extract brief context from useful sections (not code).

    Focuses on section headers and short descriptions.
    """
    lines = text.split("\n")
    result = []
    chars = 0
    in_code_block = False

    # Sections we want context from
    useful_keywords = {
        "usage",
        "example",
        "getting started",
        "quick start",
        "api",
        "overview",
        "feature",
    }

    skip_sections = {
        "changelog",
        "history",
        "contributing",
        "license",
        "citation",
        "star history",
        "acknowledgment",
    }

    include_section = False
    lines_in_section = 0

    for line in lines:
        # Track code blocks (skip them - already extracted)
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        # Check for section headers
        if line.startswith("#"):
            section_name = line.lstrip("#").strip().lower()

            # Skip unwanted sections
            if any(skip in section_name for skip in skip_sections):
                include_section = False
                continue

            # Include useful sections
            if any(kw in section_name for kw in useful_keywords):
                include_section = True
                lines_in_section = 0
                if chars + len(line) + 1 <= max_chars:
                    result.append(line)
                    chars += len(line) + 1
                continue

            include_section = False
            continue

        # Skip noise
        stripped = line.strip()
        if not stripped:
            continue
        if "![" in line and "](" in line:  # badges
            continue
        if stripped.startswith("[") and "]: http" in stripped:  # ref links
            continue
        if stripped in {"---", "***", "___"}:
            continue

        # Include content from useful sections (limit per section)
        if include_section and lines_in_section < 5:
            if chars + len(line) + 1 <= max_chars:
                result.append(line)
                chars += len(line) + 1
                lines_in_section += 1

    return "\n".join(result).strip()


def get_library_context_for_query(query: str) -> Optional[LibraryContext]:
    """
    Convenience function: detect and fetch library context from a query.

    Args:
        query: User's plugin generation query

    Returns:
        LibraryContext for the first detected library, or None
    """
    libraries = detect_library_mentions(query)

    if not libraries:
        return None

    # Fetch context for the first (primary) library mentioned
    return fetch_library_context(libraries[0])


__all__ = [
    "LibraryContext",
    "detect_library_mentions",
    "fetch_library_context",
    "get_library_context_for_query",
    "KNOWN_LIBRARIES",
]
