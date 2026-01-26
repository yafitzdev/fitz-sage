# examples/04_multi_collection.py
"""
Multiple Collections - Separate knowledge bases for different domains.

Each collection is isolated - different documents, different embeddings,
stored in separate PostgreSQL databases. Perfect for:
- Multi-tenant applications
- Domain-specific knowledge bases
- A/B testing different document sets

Requirements:
    pip install fitz-ai
    export COHERE_API_KEY="your-key"

Run:
    python examples/04_multi_collection.py
"""

import tempfile
from pathlib import Path

from fitz_ai import fitz

# =============================================================================
# Setup: Create documents for different domains
# =============================================================================

temp_dir = Path(tempfile.mkdtemp())

# Engineering docs
eng_dir = temp_dir / "engineering"
eng_dir.mkdir()
(eng_dir / "architecture.md").write_text(
    """
# System Architecture

Our system uses a microservices architecture with:
- API Gateway for routing
- PostgreSQL for persistent storage
- Redis for caching
- Kubernetes for orchestration

Deployment happens via CI/CD pipeline with automated testing.
"""
)

(eng_dir / "coding_standards.md").write_text(
    """
# Coding Standards

- Use Python 3.10+ with type hints
- Follow PEP 8 style guide
- Write unit tests for all new code
- Document public APIs with docstrings
"""
)

# HR docs
hr_dir = temp_dir / "hr"
hr_dir.mkdir()
(hr_dir / "benefits.md").write_text(
    """
# Employee Benefits

- Health insurance (medical, dental, vision)
- 401(k) with 4% company match
- 20 days PTO + 10 holidays
- Remote work flexibility
- $1,500 annual learning budget
"""
)

(hr_dir / "policies.md").write_text(
    """
# Company Policies

## Time Off
Request PTO at least 2 weeks in advance for vacations.
Sick days don't require advance notice.

## Remote Work
Employees can work remotely up to 3 days per week.
Core hours are 10am-3pm for meetings.
"""
)

# =============================================================================
# Step 1: Create separate collections
# =============================================================================

print("Creating separate knowledge bases...\n")

# Engineering collection
eng_kb = fitz(collection="engineering_docs")
eng_stats = eng_kb.ingest(str(eng_dir))
print(f"Engineering KB: {eng_stats.chunks} chunks")

# HR collection
hr_kb = fitz(collection="hr_docs")
hr_stats = hr_kb.ingest(str(hr_dir))
print(f"HR KB: {hr_stats.chunks} chunks")

# =============================================================================
# Step 2: Query each collection independently
# =============================================================================

print("\n" + "=" * 60)
print("ENGINEERING QUERIES")
print("=" * 60)

eng_questions = [
    "What database do we use?",
    "What are the coding standards?",
]

for q in eng_questions:
    print(f"\nQ: {q}")
    answer = eng_kb.ask(q)
    print(f"A: {answer.text}")

print("\n" + "=" * 60)
print("HR QUERIES")
print("=" * 60)

hr_questions = [
    "How much PTO do I get?",
    "What's the 401k match?",
]

for q in hr_questions:
    print(f"\nQ: {q}")
    answer = hr_kb.ask(q)
    print(f"A: {answer.text}")

# =============================================================================
# Step 3: Show isolation - HR questions to Engineering KB fail gracefully
# =============================================================================

print("\n" + "=" * 60)
print("COLLECTION ISOLATION")
print("=" * 60)

print("\nAsking HR question to Engineering KB:")
print("Q: What's my PTO allowance?")
answer = eng_kb.ask("What's my PTO allowance?")
print(f"A: {answer.text}")
print("(Notice: Engineering KB doesn't know about HR policies)")

# =============================================================================
# Step 4: List and manage collections
# =============================================================================

print("\n" + "=" * 60)
print("COLLECTION MANAGEMENT")
print("=" * 60)

# You can also use the CLI for collection management:
print(
    """
CLI commands for managing collections:

  fitz collections              # Interactive browser
  fitz collections list         # List all collections
  fitz collections delete NAME  # Delete a collection

Or programmatically via the vector DB:

  from fitz_ai.vector_db import get_vector_db
  vdb = get_vector_db()
  collections = vdb.list_collections()
  vdb.delete_collection("old_collection")
"""
)

# Cleanup
import shutil

shutil.rmtree(temp_dir)
print("\nDemo complete!")
