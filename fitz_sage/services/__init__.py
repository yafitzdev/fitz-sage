# fitz_sage/services/__init__.py
"""
Unified Service Layer - Single API for CLI, SDK, and REST API.

All three interfaces (CLI, SDK, API) should call this service layer.
This ensures consistent behavior and allows testing in one place.
"""

from .fitz_service import FitzService

__all__ = ["FitzService"]
