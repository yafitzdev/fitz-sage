# fitz_ai/structured/constants.py
"""
Constants for structured data storage.

Collection naming conventions and safety limits.
"""

# Collection suffixes
SCHEMA_COLLECTION_SUFFIX = "__schema"
TABLES_COLLECTION_SUFFIX = "__tables"
DERIVED_COLLECTION_SUFFIX = "__derived"

# Payload field names (prefixed with __ to avoid collision with user columns)
FIELD_TABLE = "__table"
FIELD_PRIMARY_KEY = "__pk"
FIELD_ROW_DATA = "__row"

# Safety limits
MAX_SCAN_ROWS = 10_000  # Hard error if table exceeds this
MAX_RESULT_NAMES = 50  # GROUP_CONCAT limit for names
MAX_QUERIES_PER_REQUEST = 5  # Max SQL queries LLM can generate
MAX_INDEXED_COLUMNS = 5  # Max columns to auto-index (excluding PK)

# Batch sizes
UPSERT_BATCH_SIZE = 100  # Rows per upsert call


def get_schema_collection(base: str) -> str:
    """Get schema collection name for a base collection."""
    return f"{base}{SCHEMA_COLLECTION_SUFFIX}"


def get_tables_collection(base: str) -> str:
    """Get tables collection name for a base collection."""
    return f"{base}{TABLES_COLLECTION_SUFFIX}"


def get_derived_collection(base: str) -> str:
    """Get derived collection name for a base collection."""
    return f"{base}{DERIVED_COLLECTION_SUFFIX}"


def make_row_id(table_name: str, primary_key_value: str) -> str:
    """Create a unique row ID from table name and primary key."""
    return f"{table_name}:{primary_key_value}"


__all__ = [
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
]
