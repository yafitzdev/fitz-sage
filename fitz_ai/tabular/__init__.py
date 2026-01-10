# fitz_ai/tabular/__init__.py
"""
Tabular Data Routing - Store tables as structured data, query with SQL.

Tables are extracted during ingestion and stored as JSON in schema chunk payloads.
At query time, relevant columns are pruned, an in-memory SQLite is built,
and SQL is generated and executed to return precise results.

Usage (Ingestion):
    from fitz_ai.tabular import TableExtractor

    extractor = TableExtractor()
    modified_doc, table_chunks = extractor.extract(parsed_doc)

Usage (Query):
    from fitz_ai.tabular import TableQueryStep

    step = TableQueryStep(chat=chat_client)
    augmented_chunks = step.execute(query, chunks)
"""

from .extractor import TableExtractor
from .models import ParsedTable, create_schema_chunk
from .query import TableQueryStep

__all__ = [
    "ParsedTable",
    "create_schema_chunk",
    "TableExtractor",
    "TableQueryStep",
]
