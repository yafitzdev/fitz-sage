# examples/01_quickstart.py
"""
Quickstart - The simplest way to use Fitz.

This is the 90% use case: ingest docs, ask questions, get answers with sources.

Requirements:
    pip install fitz-ai
    export COHERE_API_KEY="your-key"  # or OPENAI_API_KEY

Run:
    python examples/01_quickstart.py
"""

from fitz_ai import fitz

# =============================================================================
# Setup: Create a Fitz instance
# =============================================================================

f = fitz(collection="quickstart_demo")

# =============================================================================
# Step 1: Ingest documents
# =============================================================================

# Point at any folder - Fitz handles PDFs, DOCX, Markdown, code, etc.
print("Ingesting documents...")
stats = f.ingest("./docs")  # Change to your docs folder
print(f"Ingested {stats.chunks} chunks from {stats.documents} documents\n")

# =============================================================================
# Step 2: Ask questions
# =============================================================================

questions = [
    "What is this project about?",
    "How do I get started?",
    "What are the main features?",
]

for question in questions:
    print(f"Q: {question}")
    answer = f.ask(question)
    print(f"A: {answer.text}\n")

    # Every answer includes sources
    if answer.provenance:
        print("Sources:")
        for source in answer.provenance[:3]:  # Show top 3
            print(f"  - {source.source_id}")
        print()

# =============================================================================
# That's it! For more control, see the other examples.
# =============================================================================
