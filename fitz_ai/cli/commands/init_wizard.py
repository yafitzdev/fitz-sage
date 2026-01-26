# fitz_ai/cli/commands/init_wizard.py
"""
Init wizard - main orchestration for interactive setup.
"""

from __future__ import annotations

import typer

from fitz_ai.cli.context import CLIContext
from fitz_ai.cli.ui import RICH, console, ui
from fitz_ai.core.registry import (
    available_chunking_plugins,
    available_llm_plugins,
    available_retrieval_plugins,
    available_vector_db_plugins,
)
from fitz_ai.runtime import get_default_engine, get_engine_registry

from .init_config import (
    copy_engine_default_config,
    generate_fitz_rag_config,
    generate_global_config,
)
from .init_detector import detect_system, filter_available_plugins, load_default_config
from .init_models import get_default_model, get_default_or_first, prompt_model


def _run_fitz_rag_wizard(system, non_interactive: bool) -> str:
    """Run FitzRAG-specific configuration wizard.

    Returns:
        Generated FitzRAG config YAML string.
    """
    # Discover plugins
    all_chat = available_llm_plugins("chat")
    all_embedding = available_llm_plugins("embedding")
    all_rerank = available_llm_plugins("rerank")
    all_vision = available_llm_plugins("vision")
    all_vector_db = available_vector_db_plugins()
    all_retrieval = available_retrieval_plugins()
    all_chunkers = available_chunking_plugins()

    # Filter to available only
    avail_chat = filter_available_plugins(all_chat, "chat", system)
    avail_embedding = filter_available_plugins(all_embedding, "embedding", system)
    avail_rerank = filter_available_plugins(all_rerank, "rerank", system)
    avail_vision = filter_available_plugins(all_vision, "vision", system)
    avail_vector_db = filter_available_plugins(all_vector_db, "vector_db", system)
    avail_retrieval = all_retrieval
    avail_chunkers = all_chunkers if all_chunkers else ["simple"]

    # Validate minimum requirements
    if not avail_chat:
        ui.error("No chat plugins available!")
        ui.info("Set an API key (COHERE_API_KEY, OPENAI_API_KEY) or start Ollama.")
        raise typer.Exit(1)

    if not avail_embedding:
        ui.error("No embedding plugins available!")
        ui.info("Set an API key (COHERE_API_KEY, OPENAI_API_KEY) or start Ollama.")
        raise typer.Exit(1)

    if not avail_vector_db:
        ui.error("No vector database available!")
        ui.info("Ensure pgvector packages are installed: pip install psycopg pgvector pgserver")
        raise typer.Exit(1)

    # Load defaults from default.yaml
    default_config = load_default_config()

    # Plugin defaults
    default_chat = default_config.get("chat", {}).get("plugin_name", "cohere")
    default_embedding = default_config.get("embedding", {}).get("plugin_name", "cohere")
    default_vector_db = default_config.get("vector_db", {}).get("plugin_name", "pgvector")
    default_retrieval = default_config.get("retrieval", {}).get("plugin_name", "dense")
    default_rerank = default_config.get("rerank", {}).get("plugin_name", "cohere")

    # Chunking defaults
    default_ingest = default_config.get("ingest", {})
    default_chunking = default_ingest.get("chunking", {}).get("default", {})
    default_chunker = default_chunking.get("plugin_name", "recursive")
    default_chunk_size = default_chunking.get("kwargs", {}).get("chunk_size", 1000)
    default_chunk_overlap = default_chunking.get("kwargs", {}).get("chunk_overlap", 200)

    if non_interactive:
        # Use defaults if available
        chat_choice = get_default_or_first(avail_chat, default_chat)
        chat_model_smart = get_default_model("chat", chat_choice, "smart")
        chat_model_fast = get_default_model("chat", chat_choice, "fast")
        chat_model_balanced = get_default_model("chat", chat_choice, "balanced")
        embedding_choice = get_default_or_first(avail_embedding, default_embedding)
        embedding_model = get_default_model("embedding", embedding_choice)
        rerank_choice = get_default_or_first(avail_rerank, default_rerank) if avail_rerank else None
        rerank_model = get_default_model("rerank", rerank_choice) if rerank_choice else ""
        vector_db_choice = get_default_or_first(avail_vector_db, default_vector_db)
        retrieval_choice = get_default_or_first(avail_retrieval, default_retrieval)
        chunker_choice = default_chunker
        chunk_size = default_chunk_size
        chunk_overlap = default_chunk_overlap
        vision_choice = avail_vision[0] if avail_vision else None
        vision_model = get_default_model("vision", vision_choice) if vision_choice else ""
        parser_choice = "docling_vision" if vision_choice else "docling"
    else:
        # Interactive selection
        ui.section("Configuration")

        # Chat plugin with smart/fast/balanced model selection
        chat_choice = ui.prompt_numbered_choice(
            "Chat plugin", avail_chat, get_default_or_first(avail_chat, default_chat)
        )
        chat_model_smart = prompt_model("chat", chat_choice, "smart")
        chat_model_fast = prompt_model("chat", chat_choice, "fast")
        chat_model_balanced = prompt_model("chat", chat_choice, "balanced")

        # Embedding
        print()
        embedding_choice = ui.prompt_numbered_choice(
            "Embedding plugin",
            avail_embedding,
            get_default_or_first(avail_embedding, default_embedding),
        )
        embedding_model = prompt_model("embedding", embedding_choice)

        # Rerank provider
        rerank_choice = None
        rerank_model = ""
        if avail_rerank:
            print()
            rerank_choice = ui.prompt_numbered_choice(
                "Rerank plugin",
                avail_rerank,
                get_default_or_first(avail_rerank, default_rerank),
            )
            rerank_model = prompt_model("rerank", rerank_choice)
        else:
            print()
            ui.info("Rerank: not available (no rerank plugins detected)")

        # Vision provider
        vision_choice = None
        vision_model = ""
        if avail_vision:
            print()
            vision_choice = ui.prompt_numbered_choice(
                "Vision plugin",
                avail_vision,
                avail_vision[0],
            )
            vision_model = prompt_model("vision", vision_choice)
        else:
            print()
            ui.info("Vision: not available (no vision plugins detected)")

        # Vector DB
        print()
        vector_db_choice = ui.prompt_numbered_choice(
            "Vector database",
            avail_vector_db,
            get_default_or_first(avail_vector_db, default_vector_db),
        )

        # Retrieval
        print()
        ui.info("Retrieval plugin determines if reranking is used")
        retrieval_choice = ui.prompt_numbered_choice(
            "Retrieval strategy",
            avail_retrieval,
            get_default_or_first(avail_retrieval, default_retrieval),
        )

        # Chunking
        print()
        chunker_choice = ui.prompt_numbered_choice(
            "Chunking strategy", avail_chunkers, default_chunker
        )
        chunk_size = ui.prompt_int("  Chunk size", default_chunk_size)
        chunk_overlap = ui.prompt_int("  Chunk overlap", default_chunk_overlap)

        # Parser selection
        print()
        if avail_vision:
            parser_descs = [
                "docling_vision - VLM-powered figure description",
                "docling - Standard document parsing",
            ]
            selected = ui.prompt_numbered_choice("Parser", parser_descs, parser_descs[0])
            parser_choice = selected.split(" - ")[0]
        else:
            ui.info("Parser: docling (VLM not available)")
            parser_choice = "docling"

    # Generate config
    return generate_fitz_rag_config(
        chat=chat_choice,
        chat_model_smart=chat_model_smart,
        chat_model_fast=chat_model_fast,
        chat_model_balanced=chat_model_balanced,
        embedding=embedding_choice,
        embedding_model=embedding_model,
        rerank=rerank_choice,
        rerank_model=rerank_model,
        vector_db=vector_db_choice,
        retrieval=retrieval_choice,
        chunker=chunker_choice,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parser=parser_choice,
        vision=vision_choice,
        vision_model=vision_model,
    )


def _save_configs(
    global_config_yaml: str,
    engine_config_yaml: str | None,
    default_engine: str,
    non_interactive: bool,
) -> None:
    """Save configuration files to disk."""
    from fitz_ai.core.paths import FitzPaths

    fitz_dir = FitzPaths.workspace()
    fitz_config = FitzPaths.config()

    # Confirm overwrite if exists
    if engine_config_yaml:
        engine_config_path = FitzPaths.engine_config(default_engine)
        if (fitz_config.exists() or engine_config_path.exists()) and not non_interactive:
            if not ui.prompt_confirm("Config exists. Overwrite?", default=False):
                ui.warning("Aborted.")
                raise typer.Exit(0)

    fitz_dir.mkdir(parents=True, exist_ok=True)
    fitz_config.write_text(global_config_yaml)
    ui.success(f"Saved global config to {fitz_config}")

    if engine_config_yaml:
        engine_config_path = FitzPaths.engine_config(default_engine)
        FitzPaths.ensure_config_dir()
        engine_config_path.write_text(engine_config_yaml)
        ui.success(f"Saved engine config to {engine_config_path}")


def _show_next_steps() -> None:
    """Display next steps after successful configuration."""
    if RICH:
        console.print(
            """
[green]Your configuration is ready![/green]

Next steps:
  [cyan]fitz ingest ./docs[/cyan]          # Ingest documents
  [cyan]fitz query "your question"[/cyan]  # Query knowledge base
  [cyan]fitz doctor[/cyan]                 # Verify setup
"""
        )
    else:
        print(
            """
Your configuration is ready!

Next steps:
  fitz ingest ./docs          # Ingest documents
  fitz query "your question"  # Query knowledge base
  fitz doctor                 # Verify setup
"""
        )


def command(
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "-y",
        help="Use detected defaults without prompting.",
    ),
    show_config: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Preview config without saving.",
    ),
) -> None:
    """
    Initialize Fitz with an interactive setup wizard.

    Detects available providers (API keys, Ollama, pgvector) and
    creates a working configuration file.

    Examples:
        fitz init           # Interactive wizard
        fitz init -y        # Auto-detect and use defaults
        fitz init --show    # Preview config without saving
    """
    # Header
    ui.header("Fitz Init", "Let's configure your RAG pipeline")

    # Select default engine
    ctx = CLIContext.load()
    registry = get_engine_registry()

    if non_interactive:
        default_engine = get_default_engine()
    else:
        ui.section("Engine Setup")
        default_engine = ctx.select_engine()

    # Check if engine supports collections (FitzRAG does, others may not)
    caps = registry.get_capabilities(default_engine)

    if not caps.supports_collections:
        # Non-collection engines - use default config from registry
        global_config_yaml = generate_global_config(default_engine)
        engine_config_yaml = copy_engine_default_config(default_engine, registry)

        # Show config
        ui.section("Generated Configuration")
        ui.print("Global config (.fitz/config.yaml):", "bold")
        ui.syntax(global_config_yaml, "yaml")

        if engine_config_yaml:
            print()
            ui.print(f"Engine config (.fitz/config/{default_engine}.yaml):", "bold")
            ui.syntax(engine_config_yaml, "yaml")

        if show_config:
            return

        # Save
        ui.section("Saving")
        _save_configs(global_config_yaml, engine_config_yaml, default_engine, non_interactive)

        ui.section("Done!")
        ui.info(
            f"Use 'fitz quickstart <folder> \"question\" --engine {default_engine}' to get started."
        )
        return

    # FitzRAG and other collection-based engines
    ui.section("Detecting System")
    system = detect_system()

    # Show detection results
    ui.status(
        "Ollama",
        system.ollama.available,
        (
            f"{system.ollama.host}:{system.ollama.port}"
            if system.ollama.available
            else system.ollama.details
        ),
    )
    ui.status("pgvector", system.pgvector.available, system.pgvector.details)

    for name, key_status in system.api_keys.items():
        ui.status(name.capitalize(), key_status.available)

    # Generate configs
    global_config_yaml = generate_global_config(default_engine)

    if default_engine == "fitz_rag":
        engine_config_yaml = _run_fitz_rag_wizard(system, non_interactive)
    else:
        # For any other engine, use auto-discovery
        engine_config_yaml = copy_engine_default_config(default_engine, registry)
        if engine_config_yaml is None:
            ui.warning(f"No default config found for engine '{default_engine}'")
            engine_config_yaml = f"# {default_engine} configuration\n# Edit as needed\n"

    # Show config
    ui.section("Generated Configuration")
    ui.print("Global config (.fitz/config.yaml):", "bold")
    ui.syntax(global_config_yaml, "yaml")
    print()
    ui.print(f"Engine config (.fitz/config/{default_engine}.yaml):", "bold")
    ui.syntax(engine_config_yaml, "yaml")

    if show_config:
        return

    # Save
    ui.section("Saving")
    _save_configs(global_config_yaml, engine_config_yaml, default_engine, non_interactive)

    # Done
    ui.section("Done!")
    _show_next_steps()
