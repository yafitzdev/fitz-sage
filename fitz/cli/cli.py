# fitz/cli/cli.py

"""
Main Fitz CLI.

Goals:
- Discoverability first
- Zero magic
- No core side effects
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from fitz.core.llm.registry import LLM_REGISTRY
from fitz.pipeline.pipeline.registry import available_pipeline_plugins
from fitz.ingest.chunking.registry import CHUNKER_REGISTRY
from fitz.ingest.ingestion.registry import REGISTRY as INGEST_REGISTRY

app = typer.Typer(
    help="Fitz — local-first RAG framework",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Register sub-apps with lazy imports to avoid circular dependency issues
# ---------------------------------------------------------------------------

def _register_sub_apps():
    """Register ingest and pipeline sub-apps after module initialization."""
    # Import here to avoid circular imports at module load time
    from fitz.ingest.cli import app as ingest_app
    from fitz.pipeline.cli import app as pipeline_app

    app.add_typer(ingest_app, name="ingest")
    app.add_typer(pipeline_app, name="pipeline")


# Call immediately during module initialization
_register_sub_apps()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def check_ollama_installed() -> bool:
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# help / overview
# ---------------------------------------------------------------------------

@app.command("help")
def help_cmd() -> None:
    typer.echo()
    typer.echo("Fitz — local-first RAG framework")
    typer.echo()
    typer.echo("Common commands:")
    typer.echo("  fitz init                    Initialize a local Fitz workspace")
    typer.echo("  fitz plugins                 List available plugins")
    typer.echo("  fitz ingest run              Ingest documents into vector DB")
    typer.echo("  fitz ingest validate         Validate documents before ingestion")
    typer.echo("  fitz pipeline query          Query your knowledge base")
    typer.echo("  fitz pipeline config show    Show resolved configuration")
    typer.echo()
    typer.echo("Quick start:")
    typer.echo("  1. fitz init")
    typer.echo("  2. fitz ingest run ./docs --collection my_docs")
    typer.echo('  3. fitz pipeline query "What is this project about?"')
    typer.echo()


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command("init")
def init() -> None:
    """
    Initialize a local Fitz workspace.

    Creates .fitz/ if it does not exist and prints
    the active default setup.
    """
    root = Path.cwd()
    fitz_dir = root / ".fitz"

    if not fitz_dir.exists():
        fitz_dir.mkdir()
        typer.echo("✓ Created .fitz/")
    else:
        typer.echo("✓ .fitz/ already exists")

    typer.echo()
    typer.echo("Active defaults:")
    typer.echo("  Preset: local")
    typer.echo("  LLM: local (Ollama)")
    typer.echo("  Embedding: local")
    typer.echo("  Vector DB: local-faiss (disk)")
    typer.echo()
    typer.echo("Next steps:")
    typer.echo("  fitz ingest run ./docs --collection my_docs")
    typer.echo('  fitz pipeline query "Your question"')
    typer.echo()


# ---------------------------------------------------------------------------
# plugins
# ---------------------------------------------------------------------------

@app.command("plugins")
def plugins() -> None:
    """
    List all discovered plugins.
    """
    from fitz.core.llm.registry import available_llm_plugins
    from fitz.core.vector_db.registry import available_vector_db_plugins

    typer.echo()

    def show(title: str, plugin_type: str):
        typer.echo(f"{title}:")
        names = available_llm_plugins(plugin_type)
        if not names:
            typer.echo("  (none)")
        else:
            for name in sorted(names):
                typer.echo(f"  - {name}")
        typer.echo()

    show("LLM chat", "chat")
    show("LLM embedding", "embedding")
    show("LLM rerank", "rerank")

    # Vector DB plugins
    typer.echo("Vector DB:")
    try:
        vdb_plugins = available_vector_db_plugins()
        if not vdb_plugins:
            typer.echo("  (none)")
        else:
            for name in sorted(vdb_plugins):
                typer.echo(f"  - {name}")
    except Exception:
        # Fallback if the function doesn't exist
        typer.echo("  - local-faiss")
        typer.echo("  - qdrant")
    typer.echo()


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

@app.command("doctor")
def doctor() -> None:
    """
    Run basic diagnostics for local setup.
    """
    typer.echo()
    typer.echo("Fitz doctor")
    typer.echo()

    # Python
    typer.echo(f"✓ Python {sys.version.split()[0]}")

    # .fitz
    if Path(".fitz").exists():
        typer.echo("✓ .fitz directory present")
    else:
        typer.echo("⚠ .fitz directory missing (run: fitz init)")

    # Ollama
    if check_ollama_installed():
        typer.echo("✓ Ollama detected")
    else:
        typer.echo("⚠ Ollama not detected (local LLM unavailable)")
        typer.echo("→ To enable local LLMs: https://ollama.com")

    typer.echo()


# ---------------------------------------------------------------------------
# legacy helpers (kept for compatibility)
# ---------------------------------------------------------------------------

@app.command("setup-local")
def setup_local() -> None:
    typer.echo()
    typer.echo("Deprecated: use `fitz init` + `fitz doctor`")
    typer.echo("This command will be removed in a future release.")
    typer.echo()


@app.command("test")
def test() -> None:
    typer.echo()
    typer.echo("Deprecated: use `fitz doctor`")
    typer.echo("This command will be removed in a future release.")
    typer.echo()


if __name__ == "__main__":
    app()