# fitz_sage/ingestion/parser/plugins/plaintext.py
"""
Plain text parser for simple text files.

Handles .txt, .md, .rst and other text-based formats that don't
require complex parsing. For Markdown, extracts basic structure
(headings, code blocks).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Set

from fitz_sage.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_sage.ingestion.source.base import SourceFile

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

# Supported extensions - plain text and code files
PLAINTEXT_EXTENSIONS: Set[str] = {
    # Plain text
    ".txt",
    ".text",
    # Markdown/docs
    ".md",
    ".markdown",
    ".rst",
    # Config files
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    # Code - Python
    ".py",
    ".pyi",
    ".pyx",
    # Code - JavaScript/TypeScript
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    # Code - Web
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".svelte",
    # Code - Systems
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".cxx",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".scala",
    ".swift",
    ".m",
    # Code - Shell/Scripts
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".bat",
    ".cmd",
    # Code - Other
    ".rb",
    ".php",
    ".pl",
    ".lua",
    ".r",
    ".R",
    ".jl",
    ".ex",
    ".exs",
    ".erl",
    ".hs",
    ".ml",
    ".fs",
    ".cs",
    ".vb",
    # Data/Query
    ".sql",
    ".graphql",
    ".gql",
    # Misc
    ".xml",
    ".csv",
    ".log",
    ".gitignore",
    ".dockerignore",
    ".editorconfig",
}


@dataclass
class PlainTextParser(BaseParser):
    """
    Parser for plain text and markdown files.

    For markdown files, extracts structure (headings, code blocks, lists).
    For plain text, creates a single TEXT element.

    Example:
        parser = PlainTextParser()
        doc = parser.parse(source_file)
    """

    plugin_name: str = field(default="plaintext")
    supported_extensions: Set[str] = field(default_factory=lambda: PLAINTEXT_EXTENSIONS)
    parse_markdown_structure: bool = True
    encoding: str = "utf-8"

    def parse(self, file: SourceFile) -> ParsedDocument:
        """
        Parse a text file into structured content.

        Args:
            file: SourceFile with local_path for reading.

        Returns:
            ParsedDocument with elements.

        Raises:
            ParseError: If the file cannot be read.
        """
        # Read file content using base class helper
        content = self._read_file_text(file, encoding=self.encoding)

        # Parse based on file type
        if file.extension in {".md", ".markdown"} and self.parse_markdown_structure:
            elements = self._parse_markdown(content)
        else:
            elements = self._parse_plain_text(content)

        return ParsedDocument(
            source=file.uri,
            elements=elements,
            metadata=self._build_metadata(file),
        )

    def _parse_plain_text(self, content: str) -> List[DocumentElement]:
        """Parse plain text into a single element."""
        content = content.strip()
        if not content:
            return []

        return [
            DocumentElement(
                type=ElementType.TEXT,
                content=content,
            )
        ]

    def _parse_markdown(self, content: str) -> List[DocumentElement]:
        """Parse markdown into structured elements."""
        elements: List[DocumentElement] = []
        lines = content.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for code block
            if line.startswith("```"):
                language = line[3:].strip() or None
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if code_lines:
                    elements.append(
                        DocumentElement(
                            type=ElementType.CODE_BLOCK,
                            content="\n".join(code_lines),
                            language=language,
                        )
                    )
                i += 1
                continue

            # Check for heading
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()
                if text:
                    elements.append(
                        DocumentElement(
                            type=ElementType.HEADING,
                            content=text,
                            level=level,
                        )
                    )
                i += 1
                continue

            # Check for list item
            list_match = re.match(r"^(\s*)[-*+]\s+(.+)$", line)
            if list_match:
                indent = len(list_match.group(1))
                level = indent // 2 + 1
                text = list_match.group(2).strip()
                if text:
                    elements.append(
                        DocumentElement(
                            type=ElementType.LIST_ITEM,
                            content=text,
                            level=level,
                        )
                    )
                i += 1
                continue

            # Check for numbered list
            num_list_match = re.match(r"^(\s*)\d+\.\s+(.+)$", line)
            if num_list_match:
                indent = len(num_list_match.group(1))
                level = indent // 2 + 1
                text = num_list_match.group(2).strip()
                if text:
                    elements.append(
                        DocumentElement(
                            type=ElementType.LIST_ITEM,
                            content=text,
                            level=level,
                        )
                    )
                i += 1
                continue

            # Check for blockquote
            if line.startswith(">"):
                quote_text = line.lstrip("> ").strip()
                if quote_text:
                    elements.append(
                        DocumentElement(
                            type=ElementType.QUOTE,
                            content=quote_text,
                        )
                    )
                i += 1
                continue

            # Regular text - accumulate paragraphs
            if line.strip():
                para_lines = [line]
                i += 1
                while i < len(lines) and lines[i].strip() and not self._is_special_line(lines[i]):
                    para_lines.append(lines[i])
                    i += 1
                text = " ".join(para_line.strip() for para_line in para_lines)
                if text:
                    elements.append(
                        DocumentElement(
                            type=ElementType.TEXT,
                            content=text,
                        )
                    )
                continue

            i += 1

        return elements

    def _is_special_line(self, line: str) -> bool:
        """Check if a line is a special markdown element."""
        if line.startswith("```"):
            return True
        if re.match(r"^#{1,6}\s+", line):
            return True
        if re.match(r"^[-*+]\s+", line):
            return True
        if re.match(r"^\d+\.\s+", line):
            return True
        if line.startswith(">"):
            return True
        return False


__all__ = ["PlainTextParser", "PLAINTEXT_EXTENSIONS"]
