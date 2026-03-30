# fitz_sage/integrations/__init__.py
"""Framework integrations for Fitz Cloud optimization.

This package provides integration layers for popular RAG frameworks:
- LangChain: `pip install fitz-sage[langchain]`
- LlamaIndex: `pip install fitz-sage[llamaindex]`

All integrations share a common base optimizer that handles:
- Cloud cache lookup/store
- Client-side encryption (org_key never leaves local)
- Any embedding dimension supported
- Routing advice (Pro+ tiers)
"""

from fitz_sage.integrations.base import FitzOptimizer, OptimizationResult

__all__ = ["FitzOptimizer", "OptimizationResult"]
