# docs/roadmap/03-table-vector-gap.md
# Problem: CSV rows invisible to semantic search

## Status: Open
## Impact: Medium
## Effort: Medium

## Problem

When a CSV file is ingested, the `TableChunker` creates exactly ONE chunk: a schema
description listing column names and types. The actual row data goes exclusively to
`TableStore` for SQL-based queries. This means semantic search over row content is
impossible — the vector DB knows the table exists and what columns it has, but
can't match against any cell values.

The query analyzer correctly routes "data" queries to the table strategy, which
generates SQL. But many natural language questions about tabular data don't map
cleanly to SQL, and the router must identify the query as "data" type to even
attempt the SQL path.

## Evidence

From `ingestion/chunking/plugins/table.py`:

```python
def chunk(self, content: str, metadata: dict) -> list[Chunk]:
    # Creates ONE chunk with schema info:
    # "Table: employees.csv | Columns: name (text), age (int), dept (text) | 150 rows"
    return [schema_chunk]
    # Row data stored separately in TableStore — NOT in vector DB
```

From `engines/fitz_krag/query_analyzer.py`:

```python
# Query must be classified as "data" type to route to SQL
QueryType.DATA: {"code": 0.05, "section": 0.15, "table": 0.70, "chunk": 0.10}
```

## What Goes Wrong

1. User has `employees.csv` with 500 rows
2. Vector DB has one chunk: "Table employees.csv with columns name, age, department"
3. User asks: "Who worked in the Berlin office?" — query analyzer may classify as
   "documentation" (not "data"), bypassing SQL entirely
4. Even if classified as "data", SQL generation must guess column names and values
   without seeing the actual data distribution
5. Hybrid queries like "summarize the work history of senior engineers" can't combine
   semantic understanding with tabular filtering

## Proposed Fix

### Option A: Sample row embeddings (Recommended)

Generate 5-10 representative row descriptions and embed them alongside the schema:

```python
# For each sample row, create a natural language description:
# "Employee John Smith, age 45, department Engineering, location Berlin,
#  hired 2019-03-15, role Senior Engineer"
```

This gives semantic search anchor points into the table data. When a user asks
about "Berlin office employees", the sample row mentioning Berlin gets retrieved,
which triggers the table strategy with better context for SQL generation.

Cost: ~10 extra chunks per CSV file. Negligible for vector DB, significant for
retrieval quality.

### Option B: Column value summaries

Instead of sample rows, generate per-column value distributions:

```
Column "department": 5 unique values: Engineering (120), Sales (80),
Marketing (60), HR (40), Finance (30)
```

Cheaper than sample rows but less useful for semantic matching.

### Option C: Full row embedding (Expensive)

Embed every row as a chunk. Works for small tables (<1000 rows) but doesn't
scale. Would need row batching and dedup.

## Affected Files

| File | Change |
|------|--------|
| `ingestion/chunking/plugins/table.py` | Generate sample row chunks |
| `engines/fitz_krag/retrieval/strategies/` | Better table strategy context |
| `engines/fitz_krag/query_analyzer.py` | Possibly lower threshold for "data" classification |

## Acceptance Criteria

- [ ] CSV files produce schema chunk + sample row chunks in vector DB
- [ ] "Find employees in Berlin" retrieves relevant table context
- [ ] SQL generation has sample data for better column/value inference
- [ ] Large tables (>100K rows) don't explode chunk count
