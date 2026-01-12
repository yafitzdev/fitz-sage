# fitz_ai/retrieval/expansion/expander.py
"""
Lightweight query expansion for improved retrieval recall.

Generates query variations using:
- Synonym substitution (common technical terms)
- Acronym expansion
- Plural/singular normalization

Always active - no configuration needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryExpander:
    """
    Lightweight query expander for improved recall.

    Generates query variations without LLM calls for fast, always-on expansion.
    Uses rule-based synonym substitution and acronym expansion.
    """

    max_variations: int = 4  # Maximum additional variations (plus original)

    # Common technical synonyms (bidirectional)
    SYNONYMS: dict[str, list[str]] = field(default_factory=lambda: {
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
    })

    # Common acronym expansions
    ACRONYMS: dict[str, str] = field(default_factory=lambda: {
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
    })

    def expand(self, query: str) -> list[str]:
        """
        Generate query variations.

        Args:
            query: Original query string

        Returns:
            List of query variations (always includes original as first item)
        """
        variations: list[str] = [query]
        query_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))

        # Synonym expansion
        for word in words:
            if word in self.SYNONYMS:
                for synonym in self.SYNONYMS[word]:
                    # Create variation by replacing word with synonym
                    variation = self._replace_word(query, word, synonym)
                    if variation not in variations:
                        variations.append(variation)
                        if len(variations) > self.max_variations:
                            break
            if len(variations) > self.max_variations:
                break

        # Acronym expansion (if still have room)
        if len(variations) <= self.max_variations:
            for word in words:
                if word in self.ACRONYMS:
                    expansion = self.ACRONYMS[word]
                    variation = self._replace_word(query, word, expansion)
                    if variation not in variations:
                        variations.append(variation)
                        if len(variations) > self.max_variations:
                            break

        logger.debug(
            f"Query expansion: '{query}' → {len(variations)} variations"
        )

        return variations[:self.max_variations + 1]

    def _replace_word(self, text: str, word: str, replacement: str) -> str:
        """Replace word in text preserving case of first character."""
        # Case-insensitive word boundary replacement
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)

        def replacer(match: re.Match) -> str:
            matched = match.group(0)
            # Preserve case of first character
            if matched[0].isupper():
                return replacement.capitalize()
            return replacement

        return pattern.sub(replacer, text, count=1)
