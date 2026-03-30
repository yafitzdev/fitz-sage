# fitz_sage/api/routes/__init__.py
"""API route modules."""

from fitz_sage.api.routes.collections import router as collections_router
from fitz_sage.api.routes.health import router as health_router
from fitz_sage.api.routes.query import router as query_router

__all__ = [
    "collections_router",
    "health_router",
    "query_router",
]
