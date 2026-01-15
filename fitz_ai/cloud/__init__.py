# fitz_ai/cloud/__init__.py
"""Fitz Cloud integration for Query-Time RAG Optimizer.

This module provides client-side integration with Fitz Cloud:
- Encrypted cache lookup/store (server cannot decrypt)
- Model routing recommendations
- Tier feature checking

Key principle: org_key NEVER leaves the local environment.
"""

from fitz_ai.cloud.client import CloudClient
from fitz_ai.cloud.config import CloudConfig

__all__ = ["CloudClient", "CloudConfig"]
