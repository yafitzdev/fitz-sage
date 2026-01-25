# fitz_ai/structured/__init__.py
"""
Structured data support for shared/cloud vector DBs.

Tables and CSVs stored alongside semantic chunks, queryable via
natural language -> SQL -> natural language pipeline.
"""

from fitz_ai.structured.constants import (
    DERIVED_COLLECTION_SUFFIX,
    FIELD_PRIMARY_KEY,
    FIELD_ROW_DATA,
    FIELD_TABLE,
    MAX_INDEXED_COLUMNS,
    MAX_QUERIES_PER_REQUEST,
    MAX_RESULT_NAMES,
    MAX_SCAN_ROWS,
    SCHEMA_COLLECTION_SUFFIX,
    TABLES_COLLECTION_SUFFIX,
    UPSERT_BATCH_SIZE,
    get_derived_collection,
    get_schema_collection,
    get_tables_collection,
    make_row_id,
)
from fitz_ai.structured.derived import (
    FIELD_CONTENT,
    FIELD_DERIVED,
    FIELD_GENERATED_AT,
    FIELD_SOURCE_QUERY,
    FIELD_SOURCE_TABLE,
    FIELD_TABLE_VERSION,
    DerivedRecord,
    DerivedStore,
)
from fitz_ai.structured.executor import (
    ExecutionResult,
    QueryLimitExceededError,
    StructuredExecutor,
)
from fitz_ai.structured.formatter import (
    FormattedResult,
    ResultFormatter,
    format_multiple_results,
)
from fitz_ai.structured.ingestion import (
    MissingPrimaryKeyError,
    StructuredIngester,
    TableTooLargeError,
)
from fitz_ai.structured.router import (
    QueryRouter,
    RouteDecision,
    SemanticRoute,
    StructuredRoute,
)
from fitz_ai.structured.schema import (
    ColumnSchema,
    SchemaSearchResult,
    SchemaStore,
    TableSchema,
)
from fitz_ai.structured.sql_generator import (
    GenerationResult,
    SQLGenerator,
    SQLQuery,
)
from fitz_ai.structured.types import (
    TYPE_BOOLEAN,
    TYPE_DATE,
    TYPE_NUMBER,
    TYPE_STRING,
    coerce_value,
    infer_column_type,
    infer_type,
    is_indexable_column,
    select_indexed_columns,
)

__all__ = [
    # Constants
    "SCHEMA_COLLECTION_SUFFIX",
    "TABLES_COLLECTION_SUFFIX",
    "DERIVED_COLLECTION_SUFFIX",
    "FIELD_TABLE",
    "FIELD_PRIMARY_KEY",
    "FIELD_ROW_DATA",
    "MAX_SCAN_ROWS",
    "MAX_RESULT_NAMES",
    "MAX_QUERIES_PER_REQUEST",
    "MAX_INDEXED_COLUMNS",
    "UPSERT_BATCH_SIZE",
    "get_schema_collection",
    "get_tables_collection",
    "get_derived_collection",
    "make_row_id",
    # Schema
    "ColumnSchema",
    "TableSchema",
    "SchemaSearchResult",
    "SchemaStore",
    # Ingestion
    "StructuredIngester",
    "TableTooLargeError",
    "MissingPrimaryKeyError",
    # Router
    "QueryRouter",
    "RouteDecision",
    "SemanticRoute",
    "StructuredRoute",
    # SQL Generator
    "SQLGenerator",
    "SQLQuery",
    "GenerationResult",
    # Executor
    "StructuredExecutor",
    "ExecutionResult",
    "QueryLimitExceededError",
    # Formatter
    "ResultFormatter",
    "FormattedResult",
    "format_multiple_results",
    # Derived Storage
    "DerivedStore",
    "DerivedRecord",
    "FIELD_DERIVED",
    "FIELD_SOURCE_TABLE",
    "FIELD_SOURCE_QUERY",
    "FIELD_TABLE_VERSION",
    "FIELD_GENERATED_AT",
    "FIELD_CONTENT",
    # Types
    "TYPE_STRING",
    "TYPE_NUMBER",
    "TYPE_DATE",
    "TYPE_BOOLEAN",
    "infer_type",
    "infer_column_type",
    "coerce_value",
    "is_indexable_column",
    "select_indexed_columns",
]
