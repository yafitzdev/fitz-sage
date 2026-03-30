# fitz_sage/api/__init__.py
"""
Fitz REST API.

Provides HTTP endpoints for the Fitz RAG framework.
"""

from fitz_sage.api.app import create_app

__all__ = ["create_app"]
