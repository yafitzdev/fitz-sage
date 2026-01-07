# fitz_ai/cli/commands/quickstart.py
"""
Quickstart command - Zero-friction RAG in one command.

Usage:
    fitz quickstart                                    # Interactive prompts
    fitz quickstart ./docs "What is the refund policy?" # Direct

This command is a thin wrapper that:
1. Prompts for API key if not set
2. Creates config file if not exists (like silent `fitz init -y`)
3. Ingests documents into FAISS
4. Runs query

No separate code path - just sugar on top of the normal flow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import typer

from fitz_ai.cli.ui import display_answer, ui
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Main Command
# =============================================================================


def command(
    source: Optional[Path] = typer.Argument(
        None,
        help="Path to documents (file or directory). Will prompt if not provided.",
    ),
    question: Optional[str] = typer.Argument(
        None,
        help="Question to ask about your documents. Will prompt if not provided.",
    ),
    collection: str = typer.Option(
        "quickstart",
        "--collection",
        "-c",
        help="Collection name for vector storage",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed progress",
    ),
) -> None:
    """
    One-command RAG: ingest docs and ask a question.

    Examples:
        fitz quickstart                                      # Interactive
        fitz quickstart ./docs "What is the refund policy?"  # Direct
    """
    ui.header("Fitz Quickstart", "Zero-friction RAG")

    # =========================================================================
    # Prompt for source if not provided
    # =========================================================================

    if source is None:
        print()
        source = ui.prompt_path("Path to documents", ".")

    # Validate source exists
    if not source.exists():
        ui.error(f"Path does not exist: {source}")
        raise typer.Exit(1)

    # =========================================================================
    # Prompt for question if not provided
    # =========================================================================

    if question is None:
        question = ui.prompt_text("Question to ask")

    if not question.strip():
        ui.error("Question cannot be empty")
        raise typer.Exit(1)

    # =========================================================================
    # Run quickstart with fitz_rag + FAISS
    # =========================================================================

    _run_quickstart(source, question, collection, verbose)


# =============================================================================
# Quickstart Implementation
# =============================================================================


def _run_quickstart(source: Path, question: str, collection: str, verbose: bool) -> None:
    """Run quickstart with fitz_rag engine and FAISS."""
    engine_config_path = FitzPaths.engine_config("fitz_rag")

    # =========================================================================
    # Step 1: Auto-detect Provider
    # =========================================================================

    provider, reason, extra = _resolve_provider(engine_config_path)

    # Handle case where no provider could be resolved
    if provider is None and reason == "No provider available":
        ui.error("Could not configure an LLM provider. Exiting.")
        raise typer.Exit(1)

    # Show what we're using
    print()
    if provider is None:
        # Using existing config
        ui.success(reason)
    else:
        # Create new config for detected provider
        ui.success(f"Provider: {provider.capitalize()} ({reason})")
        _create_provider_config(engine_config_path, provider, extra)

    # =========================================================================
    # Step 3: Ingest Documents
    # =========================================================================

    ui.step(1, 3, f"Ingesting documents from {source}...")

    try:
        stats = _run_ingestion(
            source=source,
            collection=collection,
            verbose=verbose,
        )
        hierarchy_info = (
            f", {stats['hierarchy_summaries']} summaries"
            if stats.get("hierarchy_summaries")
            else ""
        )
        ui.success(
            f"Ingested {stats['documents']} documents ({stats['chunks']} chunks{hierarchy_info})"
        )

    except Exception as e:
        ui.error(f"Ingestion failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)

    # =========================================================================
    # Step 4: Run Query
    # =========================================================================

    ui.step(2, 3, "Searching and generating answer...")

    try:
        answer = _run_query(
            question=question,
            collection=collection,
            verbose=verbose,
        )

    except Exception as e:
        ui.error(f"Query failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)

    # =========================================================================
    # Step 5: Display Answer (using shared display function)
    # =========================================================================

    ui.step(3, 3, "Done!")

    display_answer(answer, show_sources=True)

    # =========================================================================
    # Step 6: Next Steps
    # =========================================================================

    ui.info("Your documents are now indexed. Run more queries with:")
    ui.info(f'  fitz query "your question" --collection {collection}')


# =============================================================================
# Provider Resolution (Auto-detection)
# =============================================================================

# Required Ollama models for quickstart
OLLAMA_REQUIRED_MODELS = {
    "embedding": ["nomic-embed-text"],
    "chat": ["llama3.2", "llama3.1", "llama3", "mistral", "gemma2"],  # Any of these works
}

# Provider configurations
PROVIDER_CONFIGS = {
    "cohere": {
        "chat": {
            "plugin_name": "cohere",
            "kwargs": {
                "models": {"smart": "command-a-03-2025", "fast": "command-r7b-12-2024"},
                "temperature": 0.2,
            },
        },
        "embedding": {"plugin_name": "cohere", "kwargs": {"model": "embed-english-v3.0"}},
        "rerank": {"enabled": False, "plugin_name": "cohere", "kwargs": {"model": "rerank-v3.5"}},
    },
    "openai": {
        "chat": {
            "plugin_name": "openai",
            "kwargs": {"models": {"smart": "gpt-4o", "fast": "gpt-4o-mini"}, "temperature": 0.2},
        },
        "embedding": {"plugin_name": "openai", "kwargs": {"model": "text-embedding-3-small"}},
        "rerank": {"enabled": False},
    },
    "ollama": {
        "chat": {
            "plugin_name": "local_ollama",
            "kwargs": {"models": {"smart": "llama3.2", "fast": "llama3.2"}},
        },
        "embedding": {"plugin_name": "local_ollama", "kwargs": {"model": "nomic-embed-text"}},
        "rerank": {"enabled": False},
    },
}


def _resolve_provider(config_path: Path) -> tuple[Optional[str], str, Optional[dict]]:
    """
    Auto-detect the best available provider.

    Resolution order:
    1. Existing config file → use it
    2. Ollama running with required models → use Ollama
    3. COHERE_API_KEY in environment → use Cohere
    4. OPENAI_API_KEY in environment → use OpenAI
    5. Nothing found → guide user to Cohere signup

    Returns:
        Tuple of (provider_name, reason_message, extra_info)
        provider_name is None if using existing config
        extra_info contains provider-specific data (e.g., ollama model names)
    """
    # -------------------------------------------------------------------------
    # 1. Check for existing config
    # -------------------------------------------------------------------------
    if config_path.exists():
        return (None, "Using existing configuration", None)

    # -------------------------------------------------------------------------
    # 2. Check for Ollama
    # -------------------------------------------------------------------------
    ollama_status = _check_ollama()
    if ollama_status["running"] and ollama_status["ready"]:
        extra = {
            "chat_model": ollama_status["chat_model"],
            "embedding_model": ollama_status["embedding_model"],
        }
        return ("ollama", f"Ollama detected ({ollama_status['chat_model']})", extra)

    # -------------------------------------------------------------------------
    # 3. Check for Cohere API key
    # -------------------------------------------------------------------------
    if os.getenv("COHERE_API_KEY"):
        return ("cohere", "COHERE_API_KEY found", None)

    # -------------------------------------------------------------------------
    # 4. Check for OpenAI API key
    # -------------------------------------------------------------------------
    if os.getenv("OPENAI_API_KEY"):
        return ("openai", "OPENAI_API_KEY found", None)

    # -------------------------------------------------------------------------
    # 5. Nothing found - guide user to get Cohere API key
    # -------------------------------------------------------------------------
    api_key = _guide_cohere_signup()
    if api_key:
        return ("cohere", "API key configured", None)

    return (None, "No provider available", None)


def _check_ollama() -> dict:
    """
    Check if Ollama is running and has the required models.

    Returns:
        Dict with keys:
        - running: bool - Is Ollama server responding?
        - ready: bool - Does it have required models?
        - chat_model: str - Full name of available chat model (e.g., "llama3.2:1b")
        - embedding_model: str - Full name of embedding model (e.g., "nomic-embed-text:latest")
        - missing: list - Models that need to be pulled
    """
    import httpx

    result = {
        "running": False,
        "ready": False,
        "chat_model": None,
        "embedding_model": None,
        "missing": [],
    }

    # Check if Ollama is running
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if response.status_code != 200:
            return result
        result["running"] = True
    except (httpx.ConnectError, httpx.TimeoutException):
        return result

    # Parse available models - keep full names with tags
    try:
        data = response.json()
        models = data.get("models", [])
        # Create mapping: base_name -> full_name (e.g., "llama3.2" -> "llama3.2:1b")
        model_map = {}
        for m in models:
            full_name = m["name"]
            base_name = full_name.split(":")[0]
            # Prefer shorter tags (e.g., "latest" over "1b-q4")
            if base_name not in model_map or len(full_name) < len(model_map[base_name]):
                model_map[base_name] = full_name
    except Exception:
        return result

    # Check for embedding model
    embedding_model = None
    for base_name in OLLAMA_REQUIRED_MODELS["embedding"]:
        if base_name in model_map:
            embedding_model = model_map[base_name]
            break

    if not embedding_model:
        result["missing"].append("nomic-embed-text")
    else:
        result["embedding_model"] = embedding_model

    # Check for chat model (any of the supported ones)
    chat_model = None
    for base_name in OLLAMA_REQUIRED_MODELS["chat"]:
        if base_name in model_map:
            chat_model = model_map[base_name]
            break

    if not chat_model:
        result["missing"].append("llama3.2 (or similar)")
    else:
        result["chat_model"] = chat_model

    # Ready if we have both
    result["ready"] = embedding_model is not None and chat_model is not None

    return result


def _guide_cohere_signup() -> Optional[str]:
    """
    Guide user through getting a free Cohere API key.

    Returns:
        The API key if successful, None otherwise.
    """
    print()
    ui.warning("No LLM provider detected")
    print()
    print("  Fitz needs an LLM to answer questions. Let's set one up!")
    print()
    print("  ╭─────────────────────────────────────────────────────────────╮")
    print("  │  Get a FREE Cohere API key (no credit card required):      │")
    print("  │                                                             │")
    print("  │  1. Go to: https://dashboard.cohere.com/api-keys           │")
    print("  │  2. Sign up with Google/GitHub (30 seconds)                │")
    print("  │  3. Copy your API key                                      │")
    print("  │  4. Paste it below                                         │")
    print("  ╰─────────────────────────────────────────────────────────────╯")
    print()

    try:
        api_key = typer.prompt("Paste your Cohere API key")
    except typer.Abort:
        return None

    if not api_key or len(api_key) < 10:
        ui.error("Invalid API key")
        return None

    # Set for this session
    os.environ["COHERE_API_KEY"] = api_key

    # Offer to save permanently
    print()
    save = typer.confirm("Save API key for future sessions?", default=True)

    if save:
        _save_api_key_to_env("COHERE_API_KEY", api_key)

    return api_key


def _save_api_key_to_env(env_var: str, api_key: str) -> None:
    """Save API key to user's shell config or environment file."""
    import platform

    home = Path.home()

    # Windows: use a .env file or tell user to set system env var
    if platform.system() == "Windows":
        env_file = home / ".fitz" / ".env"
        env_file.parent.mkdir(parents=True, exist_ok=True)

        # Read existing content
        existing = ""
        if env_file.exists():
            existing = env_file.read_text()

        # Update or add the key
        lines = existing.strip().split("\n") if existing.strip() else []
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{env_var}="):
                lines[i] = f'{env_var}="{api_key}"'
                updated = True
                break

        if not updated:
            lines.append(f'{env_var}="{api_key}"')

        env_file.write_text("\n".join(lines) + "\n")
        ui.success(f"Saved to {env_file}")
        ui.info("This will be loaded automatically by Fitz")
        return

    # Unix: try shell config files
    rc_files = [home / ".zshrc", home / ".bashrc"]

    for rc_file in rc_files:
        if rc_file.exists():
            content = rc_file.read_text()

            if env_var in content:
                ui.info(f"{env_var} already in {rc_file.name}")
                return

            with open(rc_file, "a") as f:
                f.write(f'\n# Added by fitz quickstart\nexport {env_var}="{api_key}"\n')

            ui.success(f"Saved to ~/{rc_file.name}")
            ui.info(f"Run `source ~/{rc_file.name}` or restart your terminal")
            return

    # No rc file found - create .bashrc
    bashrc = home / ".bashrc"
    with open(bashrc, "w") as f:
        f.write(f'# Created by fitz quickstart\nexport {env_var}="{api_key}"\n')

    ui.success("Created ~/.bashrc with API key")


# =============================================================================
# Config Generation
# =============================================================================


def _create_provider_config(config_path: Path, provider: str, extra: Optional[dict] = None) -> None:
    """
    Create a config file for the selected provider.

    Uses the selected provider for chat/embedding and local FAISS for vectors.

    Args:
        config_path: Path to write the config file
        provider: Provider name (cohere, openai, ollama)
        extra: Optional extra info (e.g., detected Ollama model names)
    """
    import yaml

    provider_cfg = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["cohere"])

    # For Ollama, use the actual detected model names
    if provider == "ollama" and extra:
        provider_cfg = {
            "chat": {
                "plugin_name": "local_ollama",
                "kwargs": {
                    "models": {
                        "smart": extra.get("chat_model", "llama3.2"),
                        "fast": extra.get("chat_model", "llama3.2"),
                    }
                },
            },
            "embedding": {
                "plugin_name": "local_ollama",
                "kwargs": {"model": extra.get("embedding_model", "nomic-embed-text")},
            },
            "rerank": {"enabled": False},
        }

    config = {
        "chat": provider_cfg["chat"],
        "embedding": provider_cfg["embedding"],
        "vector_db": {"plugin_name": "local_faiss", "kwargs": {}},
        "retrieval": {"plugin_name": "dense", "collection": "quickstart", "top_k": 5},
        "rerank": provider_cfg["rerank"],
        "rgs": {"enable_citations": True, "strict_grounding": True, "max_chunks": 8},
        "logging": {"level": "INFO"},
    }

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write config with header
    header = f"""\
# Fitz RAG Configuration
# Generated by: fitz quickstart
# Provider: {provider}
# Edit with: fitz config --edit

"""
    config_path.write_text(header + yaml.dump(config, default_flow_style=False, sort_keys=False))


# =============================================================================
# Ingestion (reuses existing ingest logic)
# =============================================================================


def _run_ingestion(
    source: Path,
    collection: str,
    verbose: bool = False,
) -> dict:
    """
    Run document ingestion with hierarchical summaries.

    Returns dict with 'documents', 'chunks', and 'hierarchy_summaries' counts.
    """
    from fitz_ai.core.config import load_config_dict
    from fitz_ai.engines.fitz_rag.config import (
        ChunkingRouterConfig,
        ExtensionChunkerConfig,
    )
    from fitz_ai.engines.fitz_rag.config import load_config_dict as load_default_config_dict
    from fitz_ai.ingestion.chunking.router import ChunkingRouter
    from fitz_ai.ingestion.diff.scanner import FileScanner
    from fitz_ai.ingestion.enrichment.config import HierarchyConfig
    from fitz_ai.ingestion.enrichment.hierarchy.enricher import HierarchyEnricher
    from fitz_ai.ingestion.parser import ParserRouter
    from fitz_ai.ingestion.source.base import SourceFile
    from fitz_ai.llm.registry import get_llm_plugin
    from fitz_ai.vector_db.registry import get_vector_db_plugin
    from fitz_ai.vector_db.writer import VectorDBWriter

    # Load config from engine-specific path
    config_path = FitzPaths.engine_config("fitz_rag")
    config = load_config_dict(config_path)

    # Get plugin configs
    embedding_config = config.get("embedding", {})
    vector_db_config = config.get("vector_db", {})
    chat_config = config.get("chat", {})

    # Step 1: Discover and parse documents
    if verbose:
        ui.info("Discovering and parsing documents...")

    scanner = FileScanner()
    scan_result = scanner.scan(str(source))

    if not scan_result.files:
        raise ValueError(f"No documents found in {source}")

    # Quickstart skips VLM for speed - use 'fitz ingest' for VLM figure description
    parser_router = ParserRouter(vision_client=None)
    parsed_docs = []
    for file_info in scan_result.files:
        source_file = SourceFile(
            uri=Path(file_info.path).as_uri(),
            local_path=Path(file_info.path),
            metadata={},
        )
        try:
            parsed_doc = parser_router.parse(source_file)
            if parsed_doc.full_text.strip():
                parsed_docs.append(parsed_doc)
        except Exception as e:
            if verbose:
                ui.warning(f"Failed to parse {file_info.path}: {e}")

    if not parsed_docs:
        raise ValueError(f"No documents could be parsed in {source}")

    if verbose:
        ui.info(f"Parsed {len(parsed_docs)} documents")

    # Step 2: Chunk documents
    if verbose:
        ui.info("Chunking documents...")

    # Load chunking config from user config, fall back to package defaults
    chunking_cfg = config.get("chunking") or config.get("ingest", {}).get("chunking")
    if chunking_cfg:
        default_cfg = chunking_cfg.get("default", {})
        chunking_config = ChunkingRouterConfig(
            default=ExtensionChunkerConfig(
                plugin_name=default_cfg.get("plugin_name", "recursive"),
                kwargs=default_cfg.get("kwargs", {}),
            ),
            by_extension={
                ext: ExtensionChunkerConfig(
                    plugin_name=ext_cfg.get("plugin_name", "simple"),
                    kwargs=ext_cfg.get("kwargs", {}),
                )
                for ext, ext_cfg in chunking_cfg.get("by_extension", {}).items()
            },
            warn_on_fallback=chunking_cfg.get("warn_on_fallback", False),
        )
    else:
        # Fall back to package defaults from default.yaml
        default_config = load_default_config_dict()
        default_ingest = default_config.get("ingest", {})
        default_chunking = default_ingest.get("chunking", {}).get("default", {})
        chunking_config = ChunkingRouterConfig(
            default=ExtensionChunkerConfig(
                plugin_name=default_chunking.get("plugin_name", "recursive"),
                kwargs=default_chunking.get("kwargs", {}),
            ),
            warn_on_fallback=False,
        )
    chunking_router = ChunkingRouter.from_config(chunking_config)

    chunks: List = []
    for parsed_doc in parsed_docs:
        ext = Path(parsed_doc.metadata.get("source_file", ".txt")).suffix or ".txt"
        chunker = chunking_router.get_chunker(ext)
        doc_chunks = chunker.chunk(parsed_doc)
        chunks.extend(doc_chunks)

    if not chunks:
        raise ValueError("No chunks created from documents")

    if verbose:
        ui.info(f"Created {len(chunks)} chunks")

    # Step 3: Hierarchical enrichment (group summaries + corpus summary)
    if verbose:
        ui.info("Generating hierarchical summaries...")

    # Get chat client for summarization (use fast tier)
    chat_client = get_llm_plugin(
        plugin_type="chat",
        plugin_name=chat_config.get("plugin_name", "cohere"),
        tier="fast",
        **chat_config.get("kwargs", {}),
    )

    # Create hierarchy enricher with simple mode defaults
    hierarchy_config = HierarchyConfig(enabled=True, group_by="source")
    hierarchy_enricher = HierarchyEnricher(config=hierarchy_config, chat_client=chat_client)

    # Enrich chunks (adds L1 summaries as metadata, returns chunks + L2 corpus summary)
    original_chunk_count = len(chunks)
    chunks = hierarchy_enricher.enrich(chunks)
    hierarchy_summaries = len(chunks) - original_chunk_count

    if verbose:
        ui.info(f"Generated {hierarchy_summaries} hierarchy summary chunks")

    # Step 4: Embed chunks
    if verbose:
        ui.info("Generating embeddings...")

    # get_llm_plugin returns an instance, not a class
    embedder = get_llm_plugin(
        plugin_type="embedding",
        plugin_name=embedding_config.get("plugin_name", "cohere"),
        **embedding_config.get("kwargs", {}),
    )

    vectors = []
    for chunk in chunks:
        vec = embedder.embed(chunk.content)
        vectors.append(vec)

    # Step 4: Store in vector DB
    if verbose:
        ui.info("Storing vectors...")

    vdb_plugin = get_vector_db_plugin(
        vector_db_config.get("plugin_name", "local_faiss"),
        **vector_db_config.get("kwargs", {}),
    )

    writer = VectorDBWriter(client=vdb_plugin)
    writer.upsert(collection=collection, chunks=chunks, vectors=vectors)

    return {
        "documents": len(parsed_docs),
        "chunks": original_chunk_count,
        "hierarchy_summaries": hierarchy_summaries,
    }


# =============================================================================
# Query (reuses existing query/pipeline logic)
# =============================================================================


def _run_query(
    question: str,
    collection: str,
    verbose: bool = False,
):
    """
    Run a query using the existing RAG pipeline.

    Returns an RGSAnswer object.
    """
    from fitz_ai.engines.fitz_rag.config import load_config
    from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline

    # Load typed config from engine-specific path
    config_path = FitzPaths.engine_config("fitz_rag")
    typed_config = load_config(config_path)

    # Override collection
    typed_config.retrieval.collection = collection

    if verbose:
        ui.info(f"Querying collection: {collection}")

    # Create pipeline and run
    pipeline = RAGPipeline.from_config(typed_config)
    answer = pipeline.run(question)

    return answer
