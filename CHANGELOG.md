# Changelog

All notable changes to Fitz will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2025-12-17

### 🎯 MAJOR ARCHITECTURAL REFACTOR: RAG Framework → Engine Platform

Fitz has been fundamentally restructured from a RAG framework into an engine platform. This enables support for multiple knowledge engine paradigms (Classic RAG, CLaRa, custom engines) while maintaining clean abstractions.

### ✨ Added

#### Developer Experience
- **`fitz init` Setup Wizard**: Interactive configuration with auto-detection
  - Detects available LLM providers (API keys, Ollama)
  - Detects vector databases (Qdrant, FAISS)
  - Generates working config automatically
  - Auto-selects when only one option available
  
- **`fitz doctor` Diagnostics**: Comprehensive health checks
  - Python version and dependencies
  - API key validation
  - Service connectivity (Qdrant, Ollama)
  - Configuration validation

- **Smart Qdrant Plugin**: Production-ready vector DB integration
  - Auto-creates collections on first upsert
  - Auto-detects vector dimensions
  - Handles named vs unnamed vectors automatically
  - Converts string IDs to UUIDs
  - Environment variable configuration (QDRANT_HOST, QDRANT_PORT)

- **Friendly Error Handler**: User-friendly error messages
  - Pattern matching for common errors
  - Actionable fix suggestions
  - Debug mode (FITZ_DEBUG=1) for full tracebacks

- **Progress Bars**: Rich progress display for ingestion
  - Real-time embedding progress
  - Batch writing progress
  - Time estimates
  - Quiet mode (-q) for scripts

#### Core Abstractions
- **New `fitz.core` package** with paradigm-agnostic contracts:
  - `KnowledgeEngine` protocol - The single stable abstraction all engines implement
  - `Query` dataclass - Input representation
  - `Answer` dataclass - Output representation
  - `Provenance` dataclass - Source attribution
  - `Constraints` dataclass - Query-time constraints
  - Complete exception hierarchy (`EngineError`, `QueryError`, `KnowledgeError`, `GenerationError`, etc.)

#### Engine Architecture
- **New `fitz.engines` package** organizing engines by paradigm:
  - `classic_rag/` - Classic RAG implementation (moved from root)
  - Ready for future engines (`clara/`, custom engines)
- **`ClassicRagEngine`** - Implements `KnowledgeEngine` protocol, wraps existing RAG pipeline
- **Canonical runtime** (`engines/classic_rag/runtime.py`):
  - `run_classic_rag()` - Primary entry point for Classic RAG
  - `create_classic_rag_engine()` - Factory for reusable engines

#### Universal Runtime
- **New `fitz.runtime` package** for multi-engine orchestration:
  - `EngineRegistry` - Dynamic engine discovery and registration
  - `run()` - Universal entry point supporting any engine
  - `create_engine()` - Factory for any engine type
  - `list_engines()` - Engine enumeration
  - Auto-registration system for engines

#### Shared Infrastructure
- Promoted shared services to root level:
  - `fitz.llm/` - LLM service (chat, embedding, rerank)
  - `fitz.vector_db/` - Vector database service
  - `fitz.logging/` - Logging infrastructure
  - `fitz.ingest/` - Document ingestion

### 🔄 Changed

- Cohere chat plugin now uses correct v1 API format
- Updated default model to `command-r-08-2024` (command-r-plus deprecated)
- Config now saves to correct locations automatically

#### Import Paths (BREAKING)
All import paths have changed to reflect the new architecture:

**OLD (v0.2.x)**:
```python
from fitz.pipeline.pipeline.engine import RAGPipeline
from fitz.core.llm.chat.plugins.openai import OpenAIChat
```

**NEW (v0.3.0)**:
```python
from fitz.engines.classic_rag import run_classic_rag, ClassicRagEngine
from fitz.llm.chat.plugins.openai import OpenAIChat
```

#### Public API (BREAKING)
- **Removed**: Direct `RAGPipeline` instantiation from public API
- **New**: `run_classic_rag()` is now the canonical entry point
- **New**: Universal `run()` function supports all engines

**OLD**:
```python
pipeline = RAGPipeline.from_config(config)
result = pipeline.run("What is X?")
print(result.answer)
```

**NEW**:
```python
answer = run_classic_rag("What is X?")
print(answer.text)
```

#### Answer Format (BREAKING)
- `RGSAnswer.answer` → `Answer.text`
- `RGSAnswer.sources` → `Answer.provenance`
- Source objects are now `Provenance` with standardized fields

#### CLI Commands
- **Updated**: `fitz-pipeline query` now uses new runtime
- **Added**: Support for `--max-sources` and `--filters` flags
- **Enhanced**: Better error messages using new exception hierarchy

### 🏗️ Refactored

#### Directory Structure
```
OLD (v0.2.x):
fitz/
├── pipeline/          # RAG-specific
├── retrieval/         # RAG-specific
├── generation/        # RAG-specific
└── core/              # Mixed concerns

NEW (v0.3.0):
fitz/
├── core/              # Paradigm-agnostic contracts
├── engines/
│   └── classic_rag/   # RAG implementation
│       ├── engine.py
│       ├── runtime.py
│       ├── pipeline/
│       ├── retrieval/
│       └── generation/
├── runtime/           # Multi-engine orchestration
├── llm/               # Shared LLM service
├── vector_db/         # Shared vector DB service
└── ingest/            # Shared ingestion
```

#### Module Organization
- **Moved**: All RAG-specific code → `engines/classic_rag/`
- **Promoted**: Shared services → root level
- **Eliminated**: `core/` as a catch-all package
- **Fixed**: All circular dependencies

### 🐛 Fixed
- Fixed all import path inconsistencies
- Resolved circular dependency issues
- Fixed double-nesting of modules
- Corrected model location inconsistencies
- Qdrant point ID format (string → UUID conversion)
- UTF-16 file encoding detection in local filesystem plugin
- Named vector configuration handling

### 📚 Documentation
- **New**: Architecture overview in README
- **New**: Migration guide (v0.2.x → v0.3.0)
- **New**: Quickstart examples using new API
- **Updated**: All code examples to use new imports
- **Added**: Comprehensive docstrings for all core types

### 🧪 Testing
- ✅ All existing tests updated and passing
- ✅ Import path migrations verified
- ✅ Backwards compatibility layer removed (clean break)

### ⚠️ Breaking Changes Summary

1. **Import paths changed** - All `fitz.pipeline.*`, `fitz.core.llm.*`, etc. must be updated
2. **Public API changed** - Use `run_classic_rag()` instead of `RAGPipeline.run()`
3. **Answer format changed** - `Answer.text` and `Answer.provenance` instead of old attributes
4. **CLI behavior unchanged** - Commands work the same, internal implementation changed

### 🔮 Future Compatibility

This architecture enables:
- **CLaRa engine** (citation-attributed reasoning) - Coming in v0.4.0
- **Custom engines** - Users can implement `KnowledgeEngine` protocol
- **Multi-engine applications** - Switch engines dynamically
- **Engine composition** - Combine engines for ensemble approaches

### 📦 Migration Path

See `MIGRATION.md` for detailed upgrade instructions.

**Quick migration**:
1. Update all imports (find/replace `fitz.pipeline` → `fitz.engines.classic_rag`)
2. Replace `RAGPipeline.run()` with `run_classic_rag()`
3. Update answer access: `result.answer` → `answer.text`
4. Update sources: `result.sources` → `answer.provenance`
5. Run tests

---

## [0.2.x] - Previous Releases

See previous changelog entries for v0.2.x release notes.