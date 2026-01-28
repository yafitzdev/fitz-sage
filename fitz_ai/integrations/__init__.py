# fitz_ai/integrations/__init__.py
"""Framework integrations for Fitz Cloud optimization.

This package provides integration layers for popular RAG frameworks:
- LangChain: `pip install fitz-ai[langchain]`
- LlamaIndex: `pip install fitz-ai[llamaindex]`

All integrations share a common base optimizer that handles:
- Cloud cache lookup/store
- Client-side encryption (org_key never leaves local)
- Any embedding dimension supported
- Routing advice (Pro+ tiers)
"""

from fitz_ai.integrations.base import FitzOptimizer, OptimizationResult

__all__ = ["FitzOptimizer", "OptimizationResult"]
