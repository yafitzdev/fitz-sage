# fitz_ai/cli/commands/ingest_runner.py
"""
Main ingestion orchestration for ingest command.

Contains the fitz_rag ingestion flow with progress tracking and summary display.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from fitz_ai.cli.context import CLIContext
from fitz_ai.cli.ui import RICH, console, ui
from fitz_ai.logging.logger import get_logger
from fitz_ai.runtime import get_default_engine

from .ingest_adapters import VectorDBWriterAdapter
from .ingest_config import build_chunking_router_config
from .ingest_direct import ingest_direct_text
from .ingest_engines import run_engine_specific_ingest
from .ingest_helpers import (
    detect_content_type,
    get_available_artifacts,
    is_code_project,
    is_direct_text,
    parse_artifact_selection,
    suggest_collection_name,
)

logger = get_logger(__name__)


def command(
    source: Optional[str] = typer.Argument(
        None,
        help="Source path (file/directory) or direct text. Prompts if not provided.",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses config default if not provided.",
    ),
    engine: Optional[str] = typer.Option(
        None,
        "--engine",
        "-e",
        help="Engine to use. Uses default from 'fitz engine' if not specified.",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Non-interactive mode, use defaults.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-ingest all files, ignoring state.",
    ),
    artifacts: Optional[str] = typer.Option(
        None,
        "--artifacts",
        "-a",
        help="Artifacts to generate: 'all', 'none', or comma-separated list (e.g. 'navigation_index,interface_catalog'). Interactive selection if not provided.",
    ),
) -> None:
    """
    Ingest documents into the vector database.

    Uses the default engine (set via 'fitz engine'). Override with --engine.
    Incremental by default - only processes new/changed files.
    Hierarchical summaries (L1/L2) are generated automatically.

    Examples:
        fitz ingest ./src            # Uses default engine
        fitz ingest ./src -a all     # All applicable artifacts
        fitz ingest ./src -a none    # No artifacts
        fitz ingest ./src -a navigation_index,interface_catalog  # Specific artifacts
        fitz ingest ./src -f         # Force re-ingest
        fitz ingest ./src -a all -y  # Non-interactive with all artifacts
        fitz ingest ./docs -e graphrag  # Use GraphRAG engine
        fitz ingest ./docs -e clara     # Use CLaRa engine
        fitz ingest "my boss likes red cars"  # Direct text ingestion
    """
    # =========================================================================
    # Direct Text Detection (before header to provide correct title)
    # =========================================================================

    if source is not None and is_direct_text(source):
        # Load context for collection and config
        ctx = CLIContext.load()
        config = ctx.raw_config

        # Get collection name
        if collection is None:
            collection = ctx.retrieval_collection

        # Ingest direct text and return
        ingest_direct_text(source, collection, ctx, config)
        return

    # =========================================================================
    # Header (for file/directory ingestion)
    # =========================================================================

    ui.header("Fitz Ingest", "Upload documents to vector database")

    # =========================================================================
    # Engine Selection (use default if not specified)
    # =========================================================================

    if engine is None:
        engine = get_default_engine()

    ui.info(f"Engine: {engine}")

    # Route to engine-specific ingest if not fitz_rag
    if engine != "fitz_rag":
        run_engine_specific_ingest(source, collection, engine, non_interactive)
        return

    # =========================================================================
    # fitz_rag Ingestion Flow
    # =========================================================================

    _run_fitz_rag_ingest(
        source=source,
        collection=collection,
        non_interactive=non_interactive,
        force=force,
        artifacts=artifacts,
    )


def _run_fitz_rag_ingest(
    source: Optional[str],
    collection: Optional[str],
    non_interactive: bool,
    force: bool,
    artifacts: Optional[str],
) -> None:
    """Run the fitz_rag ingestion flow."""
    from fitz_ai.ingestion.chunking.router import ChunkingRouter
    from fitz_ai.ingestion.parser import ParserRouter
    from fitz_ai.ingestion.state import IngestStateManager
    from fitz_ai.llm.registry import get_llm_plugin
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    # =========================================================================
    # Load config via CLIContext (always succeeds with defaults)
    # =========================================================================

    ctx = CLIContext.load()
    config = ctx.raw_config
    embedding_id = ctx.embedding_id
    vector_db_plugin = ctx.vector_db_plugin
    default_collection = ctx.retrieval_collection

    # Show force warning if applicable
    if force:
        ui.warning("Force mode: will re-ingest all files")
        print()

    # =========================================================================
    # Interactive Prompts
    # =========================================================================

    if non_interactive:
        if source is None:
            ui.error("Source path required in non-interactive mode.")
            ui.info("Usage: fitz ingest ./docs -y")
            raise typer.Exit(1)

        if collection is None:
            collection = default_collection

        # Auto-detect content type and configure enrichment
        content_type, _ = detect_content_type(source)
        has_chat_llm = bool(config.get("chat", {}).get("plugin_name"))

        if content_type == "codebase":
            if artifacts is None:
                available_artifacts = get_available_artifacts(has_llm=has_chat_llm)
                if available_artifacts:
                    if has_chat_llm:
                        artifacts = ",".join(name for name, _ in available_artifacts)
                    else:
                        artifacts = ",".join(
                            name for name, desc in available_artifacts if "requires LLM" not in desc
                        )
        else:
            if artifacts is None:
                artifacts = "none"

    else:
        # 1. Source path
        if source is None:
            source = ui.prompt_path("Source path", ".")

        # 2. Collection - smart selection
        if collection is None:
            collection = _prompt_collection(ctx, source, default_collection)

        # 3. Auto-detect content type and configure enrichment accordingly
        content_type, _ = detect_content_type(source)
        has_chat_llm = bool(config.get("chat", {}).get("plugin_name"))

        if content_type == "codebase":
            # CODEBASE: Enable codebase analysis artifacts
            if artifacts is None:
                available_artifacts = get_available_artifacts(has_llm=has_chat_llm)
                if available_artifacts:
                    if has_chat_llm:
                        artifacts = ",".join(name for name, _ in available_artifacts)
                    else:
                        artifacts = ",".join(
                            name for name, desc in available_artifacts if "requires LLM" not in desc
                        )
        else:
            # DOCUMENTS: Skip codebase artifacts
            if artifacts is None:
                artifacts = "none"

    ui.info(f"Collection: {collection}")
    print()

    # =========================================================================
    # Initialize components
    # =========================================================================

    # Determine total steps based on whether artifacts are enabled
    has_artifacts = artifacts != "none" and (artifacts is not None or is_code_project(source))
    total_steps = 4 if has_artifacts else 3

    try:
        # State manager
        state_manager = IngestStateManager()
        state_manager.load()

        # Parser router - uses docling_vision if configured for VLM
        chunking_cfg = config.get("chunking", {})
        default_chunking = chunking_cfg.get("default", {})
        docling_parser = default_chunking.get("parser", "docling")

        parser_router = ParserRouter(docling_parser=docling_parser)

        # Build chunking router from config
        router_config = build_chunking_router_config(config)
        chunking_router = ChunkingRouter.from_config(router_config)

        # Embedder
        embedding_kwargs = config.get("embedding", {}).get("kwargs", {})
        embedder = get_llm_plugin(
            plugin_type="embedding", plugin_name=ctx.embedding_plugin, **embedding_kwargs
        )

        # Vector DB writer
        vector_client = get_vector_db_plugin(vector_db_plugin)
        writer = VectorDBWriterAdapter(vector_client)

        # Enrichment pipeline (from config, CLI selection can override)
        enrichment_pipeline = _build_enrichment_pipeline(
            config=config,
            artifacts=artifacts,
            source=source,
        )

    except Exception as e:
        ui.error(f"Failed to initialize: {e}")
        raise typer.Exit(1)

    ui.step_done(1, total_steps, "Initializing...")

    # =========================================================================
    # Generate artifacts (if enabled)
    # =========================================================================

    current_step = 1
    artifacts_generated = 0
    artifact_errors: List[str] = []

    if has_artifacts and enrichment_pipeline is not None:
        current_step += 1
        ui.step(current_step, total_steps, "Generating artifacts...")

        artifacts_generated, artifact_errors = _generate_artifacts(
            state_manager=state_manager,
            writer=writer,
            embedder=embedder,
            parser_router=parser_router,
            chunking_router=chunking_router,
            collection=collection,
            embedding_id=embedding_id,
            enrichment_pipeline=enrichment_pipeline,
        )

    # =========================================================================
    # Run diff ingestion
    # =========================================================================

    current_step += 1
    ingest_step = current_step

    try:
        summary = _run_diff_ingestion(
            source=source,
            state_manager=state_manager,
            writer=writer,
            embedder=embedder,
            parser_router=parser_router,
            chunking_router=chunking_router,
            collection=collection,
            embedding_id=embedding_id,
            vector_db_plugin=vector_db_plugin,
            enrichment_pipeline=enrichment_pipeline,
            force=force,
            has_artifacts=has_artifacts,
            ingest_step=ingest_step,
            total_steps=total_steps,
        )

        # Add artifact stats from separate step
        summary.artifacts_generated = artifacts_generated
        summary.errors += len(artifact_errors)
        summary.error_details.extend(artifact_errors)

    except Exception as e:
        ui.error(f"Ingestion failed: {e}")
        logger.debug("Ingestion error", exc_info=True)
        raise typer.Exit(1)

    # =========================================================================
    # Summary
    # =========================================================================

    current_step += 1
    _display_summary(summary, current_step, total_steps, collection)


def _prompt_collection(ctx: CLIContext, source: str, default_collection: str) -> str:
    """Prompt for collection selection with smart defaults."""
    existing_collections = ctx.get_collections()
    suggested = suggest_collection_name(source)

    if existing_collections:
        # Show existing collections with numbers
        print()
        if RICH:
            from rich.prompt import Prompt

            console.print("[bold]Collection:[/bold]")
        else:
            print("Collection:")

        # Reorder: default first, then rest
        if default_collection in existing_collections:
            ordered = [default_collection] + [
                c for c in existing_collections if c != default_collection
            ]
        else:
            ordered = existing_collections

        for i, coll in enumerate(ordered, 1):
            if coll == default_collection:
                if RICH:
                    console.print(f"[cyan][{i}][/cyan] {coll} [dim](default)[/dim]")
                else:
                    print(f"[{i}] {coll} (default)")
            else:
                if RICH:
                    console.print(f"[cyan][{i}][/cyan] {coll}")
                else:
                    print(f"[{i}] {coll}")

        # Hint for creating new
        if RICH:
            console.print("[dim]Or type a name to create new[/dim]")
        else:
            print("Or type a name to create new")

        # Get choice - number selects existing, text creates new
        if RICH:
            from rich.prompt import Prompt

            response = Prompt.ask("Collection", default="1")
        else:
            response = input("Collection (1): ").strip() or "1"

        # Check if it's a number (select existing) or name (create new)
        try:
            idx = int(response)
            if 1 <= idx <= len(ordered):
                return ordered[idx - 1]
            else:
                # Invalid number, treat as new collection name
                return response
        except ValueError:
            # Not a number, use as new collection name
            return response
    else:
        # No collections exist - prompt for name
        return ui.prompt_text("Collection name", suggested)


def _build_enrichment_pipeline(
    config: dict,
    artifacts: Optional[str],
    source: str,
):
    """Build the enrichment pipeline from config and CLI args."""
    from fitz_ai.llm.registry import get_llm_plugin

    enrichment_cfg = config.get("enrichment", {})

    # Parse artifact selection from CLI or config
    available_artifact_names = [name for name, _ in get_available_artifacts(has_llm=False)]
    selected_artifacts = parse_artifact_selection(artifacts, available_artifact_names)

    # If artifacts were selected (via CLI or interactive), enable enrichment
    if selected_artifacts is not None and len(selected_artifacts) > 0:
        enrichment_cfg["enabled"] = True
        enrichment_cfg["artifacts"] = {"enabled": selected_artifacts}
    elif selected_artifacts is not None and len(selected_artifacts) == 0:
        # Explicitly disabled via 'none'
        enrichment_cfg["enabled"] = False

    # Enrichment is always enabled (hierarchy is always on)
    enrichment_cfg["enabled"] = True

    if not enrichment_cfg.get("enabled", False):
        return None

    from fitz_ai.ingestion.enrichment import EnrichmentConfig, EnrichmentPipeline
    from fitz_ai.ingestion.enrichment.artifacts.registry import get_artifact_registry

    enrichment_config = EnrichmentConfig.model_validate(enrichment_cfg)

    # Check if any selected artifact requires LLM
    # Hierarchy always requires LLM for summarization
    chat_client = None
    needs_llm = True  # Hierarchy is always on, always needs LLM

    # Also check artifacts that require LLM
    if selected_artifacts:
        registry = get_artifact_registry()
        needs_llm = needs_llm or any(
            registry.get_plugin(name) and registry.get_plugin(name).requires_llm
            for name in selected_artifacts
        )

    if needs_llm:
        # Get chat client from config - use "fast" tier for enrichment tasks
        chat_plugin = config.get("chat", {}).get("plugin_name", "cohere")
        chat_kwargs = config.get("chat", {}).get("kwargs", {})
        # Include user's model overrides if present
        chat_models = config.get("chat", {}).get("models")
        if chat_models:
            chat_kwargs["models"] = chat_models
        chat_client = get_llm_plugin(
            plugin_type="chat",
            plugin_name=chat_plugin,
            tier="fast",
            **chat_kwargs,
        )

    return EnrichmentPipeline(
        config=enrichment_config,
        project_root=Path(source).resolve(),
        chat_client=chat_client,
    )


def _generate_artifacts(
    state_manager,
    writer,
    embedder,
    parser_router,
    chunking_router,
    collection: str,
    embedding_id: str,
    enrichment_pipeline,
) -> tuple[int, List[str]]:
    """Generate artifacts and return (count, errors)."""
    from fitz_ai.ingestion.diff.executor import DiffIngestExecutor

    # Create executor for artifact generation
    executor = DiffIngestExecutor(
        state_manager=state_manager,
        vector_db_writer=writer,
        embedder=embedder,
        parser_router=parser_router,
        chunking_router=chunking_router,
        collection=collection,
        embedding_id=embedding_id,
        enrichment_pipeline=enrichment_pipeline,
    )

    artifacts_generated = 0
    artifact_errors: List[str] = []

    try:
        if RICH:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
            )

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[dim]{task.fields[name]}[/dim]"),
                console=console,
            )

            task_id = None

            def on_artifact_progress(current: int, total: int, name: str):
                nonlocal task_id
                if task_id is None:
                    task_id = progress.add_task("Artifacts", total=total, name="")
                progress.update(task_id, completed=current, name=name if name != "Done" else "")

            with progress:
                artifacts_generated, artifact_errors = executor.ingest_artifacts(
                    on_progress=on_artifact_progress
                )
        else:
            artifacts_generated, artifact_errors = executor.ingest_artifacts()

        if artifacts_generated > 0:
            ui.success(f"Generated {artifacts_generated} artifacts")
        else:
            ui.info("No artifacts generated")

    except Exception as e:
        ui.error(f"Artifact generation failed: {e}")
        artifact_errors.append(str(e))

    return artifacts_generated, artifact_errors


def _run_diff_ingestion(
    source: str,
    state_manager,
    writer,
    embedder,
    parser_router,
    chunking_router,
    collection: str,
    embedding_id: str,
    vector_db_plugin: str,
    enrichment_pipeline,
    force: bool,
    has_artifacts: bool,
    ingest_step: int,
    total_steps: int,
):
    """Run diff ingestion with progress tracking."""
    from fitz_ai.ingestion.diff import run_diff_ingest

    progress_shown = False

    if RICH:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
        )

        # Progress bar with step prefix: [2/3] Ingesting -------- 100%
        progress = Progress(
            TextColumn(f"[bold blue][{ingest_step}/{total_steps}][/bold blue]"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )

        task_id = None
        progress_ctx = None

        def on_progress(current: int, total: int, file_path: str):
            nonlocal task_id, progress_shown, progress_ctx
            if task_id is None:
                # Start progress bar on first callback
                progress_ctx = progress.__enter__()
                task_id = progress.add_task("Ingesting", total=total)
                progress_shown = True
            progress.update(task_id, completed=current)

        summary = run_diff_ingest(
            source=source,
            state_manager=state_manager,
            vector_db_writer=writer,
            embedder=embedder,
            parser_router=parser_router,
            chunking_router=chunking_router,
            collection=collection,
            embedding_id=embedding_id,
            vector_db_id=vector_db_plugin,
            enrichment_pipeline=enrichment_pipeline,
            force=force,
            on_progress=on_progress,
            skip_artifacts=has_artifacts,  # Skip if we already did them
        )

        # Clean up progress bar if it was started
        if progress_ctx is not None:
            progress.__exit__(None, None, None)

        # If no progress was shown (all files skipped), show inline step
        if not progress_shown:
            ui.step_done(ingest_step, total_steps, "Ingesting (all up-to-date)")
    else:
        # Simple text progress for non-Rich mode
        print(f"[{ingest_step}/{total_steps}] Ingesting...")
        last_pct = -1

        def on_progress(current: int, total: int, file_path: str):
            nonlocal last_pct, progress_shown
            progress_shown = True
            if total == 0:
                return
            pct = int(current * 100 / total)
            if pct >= last_pct + 10 or current == total:
                print(f"  {current}/{total} ({pct}%)")
                last_pct = pct

        summary = run_diff_ingest(
            source=source,
            state_manager=state_manager,
            vector_db_writer=writer,
            embedder=embedder,
            parser_router=parser_router,
            chunking_router=chunking_router,
            collection=collection,
            embedding_id=embedding_id,
            vector_db_id=vector_db_plugin,
            enrichment_pipeline=enrichment_pipeline,
            force=force,
            on_progress=on_progress,
            skip_artifacts=has_artifacts,  # Skip if we already did them
        )

    return summary


def _display_summary(summary, current_step: int, total_steps: int, collection: str) -> None:
    """Display ingestion summary."""
    ui.step(current_step, total_steps, "Summary")

    # Build stats list - only include non-zero values (except Ingested which is always shown)
    stats: List[tuple[str, str]] = []
    stats.append(("Ingested", str(summary.ingested)))
    if summary.skipped > 0:
        stats.append(("Skipped", str(summary.skipped)))
    if summary.marked_deleted > 0:
        stats.append(("Marked deleted", str(summary.marked_deleted)))
    if summary.artifacts_generated > 0:
        stats.append(("Artifacts", str(summary.artifacts_generated)))
    if summary.hierarchy_summaries > 0:
        stats.append(("Summaries", str(summary.hierarchy_summaries)))
    if summary.errors > 0:
        stats.append(("Errors", str(summary.errors)))
    stats.append(("Duration", f"{summary.duration_seconds:.1f}s"))

    if RICH:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cyan")
        for label, value in stats:
            table.add_row(f"  {label}", value)
        console.print(table)
    else:
        for label, value in stats:
            print(f"      {label}: {value}")

    if summary.errors > 0:
        print()
        ui.warning(f"{summary.errors} errors occurred:")
        for err in summary.error_details[:5]:
            ui.info(f"  * {err}")
        if len(summary.error_details) > 5:
            ui.info(f"  ... and {len(summary.error_details) - 5} more")

    print()
    ui.success(f"Collection '{collection}' ready")
