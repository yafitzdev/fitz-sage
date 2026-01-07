# Troubleshooting Guide

Common issues and solutions for Fitz.

---

## Quick Diagnostics

Run the built-in diagnostic tool:

```bash
fitz doctor           # Run all checks
fitz doctor --verbose # More details
fitz doctor --test    # Test actual connections
```

---

## Common Issues

### Config Not Found

**Error:**
```
ConfigNotFoundError: Config file not found: .fitz/config/fitz_rag.yaml
```

**Solution:**
```bash
fitz init  # Creates config interactively
```

Or with auto-init in Python:
```python
from fitz_ai import fitz
f = fitz(auto_init=True)  # Creates default config
```

---

### No API Key

**Error:**
```
AuthenticationError: API key not found
```

**Solution:**

Set the appropriate environment variable:

```bash
# Cohere
export COHERE_API_KEY="your-key-here"

# OpenAI
export OPENAI_API_KEY="your-key-here"

# Anthropic
export ANTHROPIC_API_KEY="your-key-here"
```

On Windows:
```cmd
set COHERE_API_KEY=your-key-here
```

---

### Ollama Not Running

**Error:**
```
LLMError: Cannot connect to Ollama at http://localhost:11434
```

**Solution:**

1. Start Ollama:
   ```bash
   ollama serve
   ```

2. Pull required models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. Verify it's running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

---

### FAISS Import Error

**Error:**
```
ImportError: cannot import name 'faiss' from ...
```

**Solution:**

Install FAISS:
```bash
# CPU version (recommended)
pip install faiss-cpu

# GPU version (if you have CUDA)
pip install faiss-gpu
```

---

### Windows Symlink Error

**Error:**
```
WinError 1314: A required privilege is not held by the client
```

This occurs when downloading Docling models on Windows.

**Solution:**

Fitz automatically handles this, but if you see the error:

```python
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
```

Or set before running:
```cmd
set HF_HUB_DISABLE_SYMLINKS=1
fitz ingest ./docs
```

---

### Rate Limit Error

**Error:**
```
RateLimitError: Rate limit exceeded
```

**Solution:**

1. Wait and retry (automatic backoff)
2. Reduce batch size for ingestion
3. Use a different model tier:
   ```yaml
   chat:
     kwargs:
       models:
         fast: command-r7b-12-2024  # Lighter model
   ```

---

### Empty Chunks

**Error:**
```
ValueError: No chunks created from documents
```

**Causes:**
- Documents are empty or unreadable
- Parser failed silently
- All content filtered out

**Solution:**

1. Check document contents manually
2. Try with verbose logging:
   ```bash
   fitz ingest ./docs --verbose
   ```
3. Check supported formats in [INGESTION.md](INGESTION.md)

---

### Vector Dimension Mismatch

**Error:**
```
ValueError: Vector dimension mismatch: expected 1024, got 768
```

**Cause:** Embedding model changed after initial ingestion.

**Solution:**

Clear the collection and re-ingest:
```bash
fitz collections delete my_collection
fitz ingest ./docs --collection my_collection
```

Or in Python:
```python
fitz_ai.ingest("./docs", clear_existing=True)
```

---

### No Documents Found

**Error:**
```
ValueError: No documents found in ./path
```

**Causes:**
- Wrong path
- No supported file types
- Files filtered by .gitignore patterns

**Solution:**

1. Verify path exists:
   ```bash
   ls ./path
   ```

2. Check file extensions (supported: `.pdf`, `.docx`, `.md`, `.txt`, `.py`, etc.)

3. Try with a specific file:
   ```bash
   fitz ingest ./path/specific_file.pdf
   ```

---

### Timeout Errors

**Error:**
```
TimeoutError: Request timed out after 120 seconds
```

**Solution:**

1. Check network connection
2. For large files, increase timeout in config
3. Try smaller batches:
   ```yaml
   chunking:
     default:
       kwargs:
         chunk_size: 500  # Smaller chunks
   ```

---

## Debugging

### Enable Debug Logging

```yaml
# In .fitz/config/fitz_rag.yaml
logging:
  level: DEBUG
```

Or via environment:
```bash
FITZ_LOG_LEVEL=DEBUG fitz ingest ./docs
```

### Inspect State File

Check what files are tracked:
```bash
cat .fitz/ingest_state.json | python -m json.tool
```

### Test Individual Components

```python
# Test embedding
from fitz_ai.llm.registry import get_llm_plugin
embedder = get_llm_plugin(plugin_type="embedding", plugin_name="cohere")
vector = embedder.embed("test")
print(f"Embedding dim: {len(vector)}")

# Test vector DB
from fitz_ai.vector_db.registry import get_vector_db_plugin
vdb = get_vector_db_plugin("local_faiss")
collections = vdb.list_collections()
print(f"Collections: {collections}")

# Test chat
chat = get_llm_plugin(plugin_type="chat", plugin_name="cohere")
response = chat.chat([{"role": "user", "content": "Hello"}])
print(f"Response: {response}")
```

---

## Error Reference

### Exception Hierarchy

```
EngineError (base)
├── ConfigurationError    # Config issues
├── QueryError           # Invalid query
├── KnowledgeError       # Retrieval issues
├── GenerationError      # LLM issues
├── TimeoutError         # Timeout
└── UnsupportedOperationError

APIError
├── AuthenticationError  # Bad API key
├── RateLimitError      # Rate limited
└── ModelNotFoundError  # Invalid model

ConfigError
├── ConfigNotFoundError  # Missing config
├── ConfigParseError    # Invalid YAML
└── ConfigValidationError # Schema error
```

### HTTP Status Codes (API)

| Code | Meaning |
|------|---------|
| 400 | Bad request (invalid input) |
| 401 | Authentication failed |
| 404 | Collection/resource not found |
| 429 | Rate limited |
| 500 | Internal server error |
| 501 | Feature not supported |

---

## Getting Help

1. **Run diagnostics:** `fitz doctor --verbose`
2. **Check logs:** Enable DEBUG level
3. **Report issues:** [GitHub Issues](https://github.com/yafitzdev/fitz-ai/issues)

When reporting issues, include:
- Fitz version: `pip show fitz-ai`
- Python version: `python --version`
- OS: Windows/macOS/Linux
- Full error traceback
- Output of `fitz doctor`

---

## See Also

- [CONFIG.md](CONFIG.md) - Configuration reference
- [CLI.md](CLI.md) - CLI commands
- [INGESTION.md](INGESTION.md) - Ingestion pipeline
