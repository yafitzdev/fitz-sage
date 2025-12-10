"""
fitz_rag – Retrieval-Augmented Generation Toolkit
Public API surface for v0.1.0
"""

# Pipelines
from fitz_rag.pipeline.engine import RAGPipeline
from fitz_rag.pipeline.easy import EasyRAG
from fitz_rag.pipeline.fast import FastRAG
from fitz_rag.pipeline.standard import StandardRAG
from fitz_rag.pipeline.debug import DebugRAG

# Retrieval-Guided Synthesis
from fitz_rag.generation.rgs import RGS, RGSConfig

# Core data models
from fitz_rag.models.chunk import Chunk
from fitz_rag.models.document import Document

# Factory
from fitz_rag.pipeline.engine import create_pipeline

__all__ = [
    "RAGPipeline",
    "EasyRAG",
    "FastRAG",
    "StandardRAG",
    "DebugRAG",
    "RGS",
    "RGSConfig",
    "Chunk",
    "Document",
    "create_pipeline",
]
