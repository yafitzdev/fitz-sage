# fitz_ai/sdk/__init__.py
"""
Fitz SDK - Stateful Python interface for the Fitz RAG framework.

Provides a simple two-step API for pointing at documents and asking questions.

Examples:
    >>> from fitz_ai import fitz
    >>> f = fitz()
    >>> f.point("./docs")
    >>> answer = f.query("What is quantum computing?")
    >>> print(answer.text)
"""

from .fitz import fitz

__all__ = ["fitz"]
