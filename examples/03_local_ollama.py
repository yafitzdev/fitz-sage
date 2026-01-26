# examples/03_local_ollama.py
"""
Fully Local Setup - No API keys, no cloud, complete privacy.

Fitz can run 100% locally using:
- Ollama for LLM and embeddings
- Embedded PostgreSQL (pgserver) for storage

Requirements:
    1. Install Ollama: https://ollama.ai
    2. Pull models:
       ollama pull llama3.2
       ollama pull nomic-embed-text
    3. Start Ollama:
       ollama serve

Run:
    python examples/03_local_ollama.py
"""

import tempfile
from pathlib import Path

# =============================================================================
# Step 1: Create local config
# =============================================================================

# Fitz auto-detects Ollama, but you can also configure manually
config_content = """
# Local-only configuration - no API keys needed

chat:
  plugin_name: local_ollama
  kwargs:
    models:
      smart: llama3.2
      fast: llama3.2
    base_url: http://localhost:11434

embedding:
  plugin_name: local_ollama
  kwargs:
    model: nomic-embed-text
    base_url: http://localhost:11434

# Storage is always local PostgreSQL (embedded via pgserver)
vector_db: pgvector
vector_db_kwargs:
  mode: local
"""

# Write config to temp location
temp_dir = Path(tempfile.mkdtemp())
config_path = temp_dir / "config.yaml"
config_path.write_text(config_content)

# =============================================================================
# Step 2: Use Fitz with local config
# =============================================================================

from fitz_ai import fitz

# Create instance with local config
f = fitz(collection="local_demo", config_path=config_path)

# Create some test documents
docs_dir = temp_dir / "docs"
docs_dir.mkdir()

(docs_dir / "privacy.md").write_text(
    """
# Data Privacy Policy

All data processing happens locally on your machine.
No data is sent to external servers.
Documents are stored in a local PostgreSQL database.
Embeddings are generated using local Ollama models.
"""
)

(docs_dir / "setup.md").write_text(
    """
# Local Setup Guide

1. Install Ollama from ollama.ai
2. Pull required models: llama3.2 and nomic-embed-text
3. Run 'ollama serve' to start the local server
4. Fitz will automatically use embedded PostgreSQL
"""
)

# =============================================================================
# Step 3: Ingest and query locally
# =============================================================================

print("Ingesting documents locally...")
print("(First run may take a moment as PostgreSQL initializes)\n")

try:
    stats = f.ingest(str(docs_dir))
    print(f"Ingested {stats.chunks} chunks\n")

    # Ask questions - everything runs locally
    print("Q: How is my data protected?")
    answer = f.ask("How is my data protected?")
    print(f"A: {answer.text}\n")

    print("Q: What do I need to set up for local usage?")
    answer = f.ask("What do I need to set up for local usage?")
    print(f"A: {answer.text}\n")

    print("=" * 60)
    print("SUCCESS! Everything ran locally:")
    print("  - LLM: Ollama (llama3.2)")
    print("  - Embeddings: Ollama (nomic-embed-text)")
    print("  - Storage: PostgreSQL (embedded)")
    print("  - No data left your machine")
    print("=" * 60)

except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure Ollama is running:")
    print("  1. Install from https://ollama.ai")
    print("  2. Run: ollama pull llama3.2")
    print("  3. Run: ollama pull nomic-embed-text")
    print("  4. Run: ollama serve")

# Cleanup
import shutil

shutil.rmtree(temp_dir)
