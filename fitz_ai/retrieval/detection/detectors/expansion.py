# fitz_ai/retrieval/detection/detectors/expansion.py
"""
Query expansion detector.

Generates query variations using:
- Synonym substitution
- Acronym expansion

This is kept as a dict-based detector (not LLM) because it generates
variations from a fixed dictionary, not classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from fitz_ai.logging.logger import get_logger

from ..protocol import DetectionCategory, DetectionResult, Match

logger = get_logger(__name__)

_WORD_PATTERN = re.compile(r"\b\w+\b")

# Bidirectional synonyms - each term maps to its alternatives
SYNONYMS: dict[str, list[str]] = {
    # CRUD operations
    "delete": ["remove", "erase"],
    "remove": ["delete", "erase"],
    "create": ["add", "make", "generate"],
    "add": ["create", "insert"],
    "update": ["modify", "change", "edit"],
    "modify": ["update", "change", "edit"],
    "get": ["retrieve", "fetch", "obtain"],
    "retrieve": ["get", "fetch"],
    "fetch": ["get", "retrieve"],
    # Status/state
    "error": ["failure", "exception", "issue"],
    "failure": ["error", "exception"],
    "issue": ["problem", "error", "bug"],
    "bug": ["issue", "defect", "problem"],
    # Actions
    "start": ["begin", "launch", "initiate"],
    "stop": ["end", "halt", "terminate"],
    "run": ["execute", "perform"],
    "execute": ["run", "perform"],
    "install": ["setup", "deploy"],
    "setup": ["install", "configure"],
    "configure": ["setup", "set up"],
    # Common terms
    "file": ["document", "doc"],
    "document": ["file", "doc"],
    "folder": ["directory", "dir"],
    "directory": ["folder", "dir"],
    "user": ["account", "member"],
    "function": ["method", "procedure"],
    "method": ["function", "procedure"],
    "class": ["type", "object"],
    "list": ["array", "collection"],
    "array": ["list", "collection"],
    # Technical
    "api": ["endpoint", "interface"],
    "endpoint": ["api", "route"],
    "database": ["db", "datastore"],
    "db": ["database", "datastore"],
    "server": ["backend", "service"],
    "client": ["frontend", "app"],
    "request": ["call", "query"],
    "response": ["reply", "result"],
    # States
    "enable": ["activate", "turn on"],
    "disable": ["deactivate", "turn off"],
    "active": ["enabled", "on"],
    "inactive": ["disabled", "off"],
}

# Acronym expansions (one-way)
ACRONYMS: dict[str, str] = {
    "api": "application programming interface",
    "ui": "user interface",
    "ux": "user experience",
    "db": "database",
    "sql": "structured query language",
    "html": "hypertext markup language",
    "css": "cascading style sheets",
    "js": "javascript",
    "ts": "typescript",
    "url": "uniform resource locator",
    "http": "hypertext transfer protocol",
    "https": "hypertext transfer protocol secure",
    "json": "javascript object notation",
    "xml": "extensible markup language",
    "csv": "comma separated values",
    "pdf": "portable document format",
    "id": "identifier",
    "auth": "authentication",
    "config": "configuration",
    "env": "environment",
    "dev": "development",
    "prod": "production",
    "repo": "repository",
    "pr": "pull request",
    "ci": "continuous integration",
    "cd": "continuous deployment",
    "k8s": "kubernetes",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "vm": "virtual machine",
    "os": "operating system",
    "cpu": "central processing unit",
    "gpu": "graphics processing unit",
    "ram": "random access memory",
    "ssd": "solid state drive",
    "hdd": "hard disk drive",
    "iot": "internet of things",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "llm": "large language model",
    "rag": "retrieval augmented generation",
}

MAX_VARIATIONS = 4


@dataclass
class ExpansionDetector:
    """Generates query expansions using synonyms and acronyms."""

    _max_variations: int = field(default=MAX_VARIATIONS, init=False)

    @property
    def category(self) -> DetectionCategory:
        return DetectionCategory.EXPANSION

    def detect(self, query: str) -> DetectionResult[None]:
        """Detect expandable terms and generate variations."""
        variations = self._expand_query(query)

        # If only original query, nothing to expand
        if len(variations) <= 1:
            return DetectionResult.not_detected(self.category)

        # Get matches for metadata
        matches = self._find_matches(query)

        logger.debug(f"Query expansion: '{query}' -> {len(variations)} variations")

        return DetectionResult(
            detected=True,
            category=self.category,
            confidence=1.0,
            intent=None,
            matches=matches,
            metadata={
                "original": query,
                "variation_count": len(variations) - 1,
            },
            transformations=variations[1:],  # Exclude original
        )

    def _get_expansions(self, word: str) -> list[str]:
        """Get all expansions for a word."""
        word_lower = word.lower()
        expansions: list[str] = []

        if word_lower in SYNONYMS:
            expansions.extend(SYNONYMS[word_lower])

        if word_lower in ACRONYMS:
            expansions.append(ACRONYMS[word_lower])

        return expansions

    def _replace_word(self, query: str, word: str, replacement: str) -> str:
        """Replace a word in query, preserving case of first letter."""
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        match = pattern.search(query)

        if not match:
            return query

        original = match.group(0)

        # Preserve case of first letter
        if original[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]

        return pattern.sub(replacement, query, count=1)

    def _expand_query(self, query: str) -> list[str]:
        """Generate query variations by expanding words."""
        variations = [query]
        words = _WORD_PATTERN.findall(query.lower())

        for word in words:
            expansions = self._get_expansions(word)
            if expansions and len(variations) < self._max_variations + 1:
                # Create variation with first expansion
                variation = self._replace_word(query, word, expansions[0])
                if variation not in variations:
                    variations.append(variation)

        return variations[: self._max_variations + 1]

    def _find_matches(self, query: str) -> list[Match]:
        """Find dictionary matches in query."""
        words = _WORD_PATTERN.findall(query.lower())
        matches: list[Match] = []

        for word in words:
            word_lower = word.lower()

            # Check synonyms
            if word_lower in SYNONYMS:
                pos = query.lower().find(word_lower)
                matches.append(
                    Match(
                        text=word,
                        pattern_name="synonym",
                        start=pos,
                        end=pos + len(word) if pos >= 0 else 0,
                        groups={"expansions": ",".join(SYNONYMS[word_lower])},
                        confidence=1.0,
                    )
                )

            # Check acronyms
            if word_lower in ACRONYMS:
                pos = query.lower().find(word_lower)
                matches.append(
                    Match(
                        text=word,
                        pattern_name="acronym",
                        start=pos,
                        end=pos + len(word) if pos >= 0 else 0,
                        groups={"expansion": ACRONYMS[word_lower]},
                        confidence=1.0,
                    )
                )

        return matches
