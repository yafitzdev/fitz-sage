# fitz_ai/tabular/__init__.py
"""
Tabular Data Routing - Store tables as structured data, query with SQL.

Two modes of table storage:

1. **Embedded tables** (from documents):
   Tables extracted from markdown, PDF, etc. are stored as JSON in chunk payloads.
   Small and quick - data lives in vector DB.

2. **Standalone table files** (CSV, TSV):
   Tables from files are stored in TableStore (SQLite local, Qdrant team mode).
   Compressed with gzip for efficient storage. Schema chunk in vector DB points
   to table_id, actual data fetched at query time.

Usage (Ingestion - embedded tables):
    from fitz_ai.tabular import TableExtractor

    extractor = TableExtractor()
    modified_doc, table_chunks = extractor.extract(parsed_doc)

Usage (Ingestion - CSV files):
    from fitz_ai.tabular.parser import parse_csv, can_parse
    from fitz_ai.tabular.store import get_table_store
    from fitz_ai.tabular.models import create_schema_chunk_for_stored_table

    if can_parse(file_path):
        parsed = parse_csv(file_path)
        store = get_table_store(collection)
        hash = store.store(parsed.table_id, parsed.columns, parsed.rows, str(file_path))
        chunk = create_schema_chunk_for_stored_table(
            table_id=parsed.table_id,
            columns=parsed.columns,
            row_count=parsed.row_count,
            source_file=str(file_path),
            table_hash=hash,
            sample_rows=parsed.rows[:3],
        )

Usage (Query):
    from fitz_ai.tabular import TableQueryStep

    step = TableQueryStep(chat=chat_client, table_store=store)
    augmented_chunks = step.execute(query, chunks)
"""

from fitz_ai.tabular.extractor import TableExtractor
from fitz_ai.tabular.models import (
    ParsedTable,
    create_schema_chunk,
    create_schema_chunk_for_stored_table,
)
from fitz_ai.tabular.query import TableQueryStep

__all__ = [
    "ParsedTable",
    "create_schema_chunk",
    "create_schema_chunk_for_stored_table",
    "TableExtractor",
    "TableQueryStep",
]
