# Unified Storage with PostgreSQL + pgvector

> Why fitz-ai uses PostgreSQL instead of a dedicated vector database.

---

## TL;DR

Fitz uses **PostgreSQL + pgvector** for all storage—vectors, metadata, and structured tables. This means:

- **One database** for everything (no FAISS + SQLite + Qdrant juggling)
- **Full SQL** for table queries (joins, aggregations, GROUP BY)
- **Zero infrastructure** for local use (embedded via `pgserver`)
- **Same code path** whether you're running locally or in production

---

## The Decision

### Why Not a Dedicated Vector Database?

Dedicated vector databases (Pinecone, Weaviate, Qdrant, Milvus) are optimized for one thing: vector similarity search. They're excellent at scale—billions of vectors, millisecond latencies, distributed deployments.

But for fitz-ai's use case, they introduce unnecessary complexity:

| Concern | Dedicated Vector DB | PostgreSQL + pgvector |
|---------|--------------------|-----------------------|
| **Deployment** | Separate service to run | Embedded or existing Postgres |
| **Structured data** | Hack it or use another DB | Native SQL |
| **Hybrid queries** | Limited or none | Full SQL + vectors in one query |
| **Local development** | Docker or cloud | `pip install` (pgserver) |
| **Maintenance** | Two systems | One system |

### The Performance Question

> "Isn't pgvector slower than specialized vector databases?"

Yes—by 2-5x for pure vector search. But here's what actually matters:

| Operation | Dedicated VectorDB | pgvector | Impact on Query |
|-----------|-------------------|----------|-----------------|
| Vector search | 5ms | 15ms | +10ms |
| LLM generation | 500-2000ms | 500-2000ms | 0ms |
| **Total query time** | 505-2005ms | 515-2015ms | **+0.5-2%** |

Vector search is **less than 1% of total query time**. The LLM dominates. Optimizing vector search from 5ms to 15ms is imperceptible to users.

For fitz-ai's target scale (<10M vectors per collection), pgvector with HNSW indexing provides:
- **99% recall** (same as dedicated DBs)
- **<50ms latency** at 1M vectors
- **Zero maintenance** (no index retraining)

### What We Gained

By choosing PostgreSQL, fitz-ai gets capabilities that would require significant engineering with a dedicated vector DB:

**1. Real SQL for Structured Data**

```sql
-- This "just works" on ingested CSVs
SELECT product, AVG(price)
FROM sales_data
WHERE region = 'EMEA'
GROUP BY product
ORDER BY AVG(price) DESC;
```

With a vector-only database, you'd need to:
- Store tables as vectors (losing SQL capabilities)
- Run a separate SQL database
- Sync data between systems

**2. Hybrid Queries**

```sql
-- Vector similarity + metadata filtering in one query
SELECT * FROM chunks
WHERE source_file LIKE '%.py'
  AND created_at > '2024-01-01'
ORDER BY embedding <=> query_vector
LIMIT 10;
```

**3. Transactional Consistency**

Vectors, metadata, and tables are in one transaction. No eventual consistency issues, no sync problems.

**4. Zero-Friction Local Development**

```bash
pip install fitz-ai  # Includes pgserver
fitz quickstart ./docs "What is our refund policy?"
# PostgreSQL starts automatically, invisibly
```

No Docker. No cloud accounts. No configuration.

---

## How It Works

### Local Mode (Default)

Fitz uses [pgserver](https://github.com/orm011/pgserver)—a pip-installable embedded PostgreSQL with pgvector included.

```
~/.fitz/
└── pgdata/           # PostgreSQL data directory (auto-managed)
    ├── base/         # Database files
    ├── pg_wal/       # Write-ahead log
    └── ...
```

- **First query**: ~5 seconds (PostgreSQL initializes)
- **Subsequent queries**: <1 second (server stays warm)
- **Disk usage**: ~50MB base + your data

### External Mode (Production)

For production or shared deployments, point fitz at any PostgreSQL 14+ instance with pgvector:

```yaml
# ~/.fitz/config/fitz_rag.yaml
vector_db: pgvector
vector_db_kwargs:
  mode: external
  connection_string: postgresql://user:pass@host:5432/mydb
```

Same code, same behavior—just a different PostgreSQL instance.

---

## Technical Details

### Schema

Each collection gets its own database with two core tables:

```sql
-- Vector chunks (documents)
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    vector vector(768),           -- pgvector type
    payload JSONB NOT NULL,       -- metadata, content, source
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast similarity search
CREATE INDEX chunks_vector_idx ON chunks
USING hnsw (vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search index
CREATE INDEX chunks_content_idx ON chunks
USING gin (to_tsvector('english', payload->>'content'));
```

```sql
-- Structured tables (from CSVs)
CREATE TABLE tbl_{table_name} (
    _row_idx INTEGER,
    _source_file TEXT,
    -- Dynamic columns from CSV headers
    column1 TEXT,
    column2 TEXT,
    ...
);
```

### HNSW Index

Fitz uses HNSW (Hierarchical Navigable Small World) indexing:

| Property | Value | Why |
|----------|-------|-----|
| **Recall** | ~99% | Same as dedicated vector DBs |
| **Maintenance** | Zero | No periodic retraining needed |
| **Updates** | Incremental | Add/delete without rebuilding |
| **Parameters** | m=16, ef_construction=64 | Balanced for typical workloads |

### Hybrid Search

Fitz combines vector similarity with full-text search using Reciprocal Rank Fusion:

```sql
-- Simplified hybrid search
WITH vector_results AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY vector <=> $query) as rank
    FROM chunks LIMIT 100
),
text_results AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(...) DESC) as rank
    FROM chunks WHERE to_tsvector(...) @@ plainto_tsquery($text)
    LIMIT 100
)
SELECT id,
    1/(60 + v.rank) * 0.7 + 1/(60 + t.rank) * 0.3 as score
FROM vector_results v FULL JOIN text_results t USING (id)
ORDER BY score DESC;
```

---

## Trade-offs We Accepted

| Trade-off | Why It's Acceptable |
|-----------|---------------------|
| **5s cold start** | Only on first query of session; server stays warm |
| **2-5x slower vector search** | <1% of total query time; LLM dominates |
| **~15MB dependency** | Smaller than most ML libraries |
| **External process** | Managed automatically by pgserver |

---

## When to Use External PostgreSQL

Use the embedded pgserver (default) for:
- Local development
- Single-user deployments
- Prototyping

Use external PostgreSQL for:
- Production deployments
- Multi-user access
- Existing PostgreSQL infrastructure
- Managed services (AWS RDS, GCP Cloud SQL, etc.)

---

## Summary

Fitz chose PostgreSQL + pgvector because:

1. **Simplicity** > marginal performance gains
2. **Full SQL** > vector-only limitations
3. **One system** > multiple databases to sync
4. **Zero friction** > infrastructure requirements

For fitz-ai's scale and use case, this is the right trade-off. If you're building a system that needs to search billions of vectors with sub-millisecond latency, a dedicated vector database makes sense. For knowledge bases, codebases, and document collections—PostgreSQL is more than enough.

---

*See also: [Configuration Guide](../CONFIG.md) for pgvector settings*
