# fitz/cli/init.py
"""
Interactive setup wizard for Fitz.

Detects available providers and creates a working configuration.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

# Rich for pretty output (optional, falls back gracefully)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# =============================================================================
# Detection Functions
# =============================================================================

@dataclass
class ProviderStatus:
    """Status of a provider/service."""
    name: str
    available: bool
    details: str = ""
    env_var: str = ""


def detect_api_keys() -> dict[str, ProviderStatus]:
    """Detect which API keys are set."""
    providers = {}

    # Cohere
    cohere_key = os.getenv("COHERE_API_KEY")
    providers["cohere"] = ProviderStatus(
        name="Cohere",
        available=bool(cohere_key),
        details=f"Key: {cohere_key[:8]}..." if cohere_key else "Not set",
        env_var="COHERE_API_KEY"
    )

    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    providers["openai"] = ProviderStatus(
        name="OpenAI",
        available=bool(openai_key),
        details=f"Key: {openai_key[:8]}..." if openai_key else "Not set",
        env_var="OPENAI_API_KEY"
    )

    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    providers["anthropic"] = ProviderStatus(
        name="Anthropic",
        available=bool(anthropic_key),
        details=f"Key: {anthropic_key[:8]}..." if anthropic_key else "Not set",
        env_var="ANTHROPIC_API_KEY"
    )

    return providers


def detect_ollama() -> ProviderStatus:
    """Detect if Ollama is running."""
    try:
        # Check if ollama command exists
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse models
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = [line.split()[0] for line in lines if line.strip()]
            model_str = ", ".join(models[:3])
            if len(models) > 3:
                model_str += f" (+{len(models) - 3} more)"
            return ProviderStatus(
                name="Ollama",
                available=True,
                details=f"Models: {model_str}" if models else "No models installed"
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass

    return ProviderStatus(
        name="Ollama",
        available=False,
        details="Not installed or not running"
    )


def detect_qdrant() -> ProviderStatus:
    """Detect if Qdrant is accessible."""
    import urllib.request
    import urllib.error

    # Check common locations - prioritize env var
    env_host = os.getenv("QDRANT_HOST")
    hosts_to_try = []
    if env_host:
        hosts_to_try.append(env_host)
    hosts_to_try.extend(["localhost", "127.0.0.1", "192.168.178.2"])  # Common locations

    port = int(os.getenv("QDRANT_PORT", "6333"))

    for host in hosts_to_try:
        try:
            url = f"http://{host}:{port}/collections"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    import json
                    data = json.loads(response.read())
                    collections = data.get("result", {}).get("collections", [])
                    col_names = [c.get("name", "?") for c in collections]
                    col_str = ", ".join(col_names[:3]) if col_names else "none"
                    if len(col_names) > 3:
                        col_str += f" (+{len(col_names) - 3})"
                    return ProviderStatus(
                        name="Qdrant",
                        available=True,
                        details=f"At {host}:{port} (collections: {col_str})"
                    )
        except Exception:
            continue

    return ProviderStatus(
        name="Qdrant",
        available=False,
        details="Not found (tried localhost:6333)"
    )


def check_faiss_available() -> ProviderStatus:
    """Check if FAISS is installed."""
    try:
        import faiss
        return ProviderStatus(
            name="FAISS",
            available=True,
            details="Installed (local vector DB)"
        )
    except ImportError:
        return ProviderStatus(
            name="FAISS",
            available=False,
            details="Not installed (pip install faiss-cpu)"
        )


# =============================================================================
# Config Generation
# =============================================================================

def generate_config(
        llm_provider: str,
        embedding_provider: str,
        vector_db: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection: str = "default",
        enable_rerank: bool = False,
) -> str:
    """Generate YAML config based on user choices."""

    # LLM configs
    llm_configs = {
        "cohere": """llm:
  plugin_name: cohere
  kwargs:
    model: command-r-08-2024
    temperature: 0.2""",
        "openai": """llm:
  plugin_name: openai
  kwargs:
    model: gpt-4o-mini
    temperature: 0.2""",
        "anthropic": """llm:
  plugin_name: anthropic
  kwargs:
    model: claude-sonnet-4-20250514
    temperature: 0.2""",
        "ollama": """llm:
  plugin_name: ollama
  kwargs:
    model: llama3.2
    temperature: 0.2""",
    }

    # Embedding configs
    embedding_configs = {
        "cohere": """embedding:
  plugin_name: cohere
  kwargs:
    model: embed-english-v3.0""",
        "openai": """embedding:
  plugin_name: openai
  kwargs:
    model: text-embedding-3-small""",
        "ollama": """embedding:
  plugin_name: ollama
  kwargs:
    model: nomic-embed-text""",
    }

    # Vector DB configs
    vector_db_configs = {
        "qdrant": f"""vector_db:
  plugin_name: qdrant
  kwargs:
    host: "{qdrant_host}"
    port: {qdrant_port}""",
        "faiss": """vector_db:
  plugin_name: local-faiss
  kwargs: {}""",
    }

    # Build full config
    config = f"""# Fitz RAG Configuration
# Generated by: fitz init
# 
# Edit this file to customize your setup.
# Documentation: https://github.com/yafitzdev/fitz

# =============================================================================
# LLM (Chat) Configuration
# =============================================================================
{llm_configs.get(llm_provider, llm_configs['cohere'])}

# =============================================================================
# Embedding Configuration  
# =============================================================================
{embedding_configs.get(embedding_provider, embedding_configs['cohere'])}

# =============================================================================
# Vector Database Configuration
# =============================================================================
{vector_db_configs.get(vector_db, vector_db_configs['qdrant'])}

# =============================================================================
# Retriever Configuration
# =============================================================================
retriever:
  plugin_name: dense
  collection: {collection}
  top_k: 5

# =============================================================================
# Reranker (improves retrieval quality)
# =============================================================================
rerank:
  enabled: {str(enable_rerank).lower()}
  plugin_name: cohere
  kwargs:
    model: rerank-english-v3.0

# =============================================================================
# RGS (Retrieval-Guided Synthesis)
# =============================================================================
rgs:
  enable_citations: true
  strict_grounding: true
  max_chunks: 8
  include_query_in_context: true
  source_label_prefix: S

# =============================================================================
# Logging
# =============================================================================
logging:
  level: INFO
"""
    return config


# =============================================================================
# Display Helpers
# =============================================================================

def print_header(text: str) -> None:
    """Print a header."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold blue]{text}[/bold blue]")
    else:
        print(f"\n{'=' * 60}")
        print(text)
        print('=' * 60)


def print_status(name: str, available: bool, details: str = "") -> None:
    """Print status of a provider."""
    if RICH_AVAILABLE:
        icon = "✓" if available else "✗"
        color = "green" if available else "red"
        console.print(f"  [{color}]{icon}[/{color}] {name}: {details}")
    else:
        icon = "✓" if available else "✗"
        print(f"  {icon} {name}: {details}")


def prompt_choice(prompt: str, choices: list[str], default: str = None) -> str:
    """Prompt user for a choice."""
    choices_str = "/".join(choices)
    if default:
        choices_str += f" (default: {default})"

    while True:
        if RICH_AVAILABLE:
            response = Prompt.ask(f"{prompt} [{choices_str}]", default=default or "")
        else:
            response = input(f"{prompt} [{choices_str}]: ").strip()
            if not response and default:
                response = default

        if response in choices:
            return response

        print(f"Invalid choice. Please enter one of: {', '.join(choices)}")


def prompt_confirm(prompt: str, default: bool = True) -> bool:
    """Prompt user for yes/no."""
    if RICH_AVAILABLE:
        return Confirm.ask(prompt, default=default)
    else:
        default_str = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes")


# =============================================================================
# Main Command
# =============================================================================

def command(
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            "-y",
            help="Use detected defaults without prompting",
        ),
        show_config: bool = typer.Option(
            False,
            "--show-config",
            help="Only show what config would be generated",
        ),
) -> None:
    """
    Initialize Fitz with an interactive setup wizard.

    Detects available providers (API keys, Ollama, Qdrant) and
    creates a working configuration file.

    Examples:
        fitz init              # Interactive wizard
        fitz init -y           # Auto-detect and use defaults
        fitz init --show-config  # Preview config without saving
    """

    # Header
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold]🔧 Fitz Setup Wizard[/bold]\n"
            "Let's configure your RAG pipeline!",
            border_style="blue"
        ))
    else:
        print("\n" + "=" * 60)
        print("🔧 Fitz Setup Wizard")
        print("Let's configure your RAG pipeline!")
        print("=" * 60)

    # ==========================================================================
    # Detection Phase
    # ==========================================================================

    print_header("Checking LLM Providers...")

    api_keys = detect_api_keys()
    ollama = detect_ollama()

    for provider in api_keys.values():
        print_status(provider.name, provider.available, provider.details)
    print_status(ollama.name, ollama.available, ollama.details)

    print_header("Checking Vector Databases...")

    qdrant = detect_qdrant()
    faiss = check_faiss_available()

    print_status(qdrant.name, qdrant.available, qdrant.details)
    print_status(faiss.name, faiss.available, faiss.details)

    # ==========================================================================
    # Determine Best Defaults
    # ==========================================================================

    # LLM priority: Cohere > OpenAI > Anthropic > Ollama
    llm_default = "ollama"  # fallback
    if api_keys["cohere"].available:
        llm_default = "cohere"
    elif api_keys["openai"].available:
        llm_default = "openai"
    elif api_keys["anthropic"].available:
        llm_default = "anthropic"
    elif ollama.available:
        llm_default = "ollama"

    # Embedding priority: same provider as LLM if possible
    embedding_default = llm_default
    if embedding_default == "anthropic":
        # Anthropic doesn't have embeddings, fall back
        embedding_default = "cohere" if api_keys["cohere"].available else "openai"

    # Vector DB priority: Qdrant > FAISS
    vector_db_default = "qdrant" if qdrant.available else "faiss"

    # ==========================================================================
    # User Choices (if interactive)
    # ==========================================================================

    if non_interactive:
        llm_choice = llm_default
        embedding_choice = embedding_default
        vector_db_choice = vector_db_default
        collection_name = "default"
        enable_rerank = False
    else:
        print_header("Configuration Options")

        # LLM choice
        available_llms = []
        if api_keys["cohere"].available:
            available_llms.append("cohere")
        if api_keys["openai"].available:
            available_llms.append("openai")
        if api_keys["anthropic"].available:
            available_llms.append("anthropic")
        if ollama.available:
            available_llms.append("ollama")

        if not available_llms:
            if RICH_AVAILABLE:
                console.print("\n[red]❌ No LLM providers available![/red]")
                console.print("Please either:")
                console.print("  1. Set an API key: export COHERE_API_KEY=your-key")
                console.print("  2. Install Ollama: https://ollama.com")
            else:
                print("\n❌ No LLM providers available!")
                print("Please either:")
                print("  1. Set an API key: export COHERE_API_KEY=your-key")
                print("  2. Install Ollama: https://ollama.com")
            raise typer.Exit(code=1)

        # Auto-select if only one option
        if len(available_llms) == 1:
            llm_choice = available_llms[0]
            print(f"\n✓ Using {llm_choice} for LLM (only available provider)")
        else:
            print(f"\nAvailable LLM providers: {', '.join(available_llms)}")
            llm_choice = prompt_choice("Select LLM provider", available_llms, llm_default)

        # Embedding choice
        available_embeddings = []
        if api_keys["cohere"].available:
            available_embeddings.append("cohere")
        if api_keys["openai"].available:
            available_embeddings.append("openai")
        if ollama.available:
            available_embeddings.append("ollama")

        if llm_choice in available_embeddings:
            embedding_default = llm_choice
        elif available_embeddings:
            embedding_default = available_embeddings[0]

        # Auto-select if only one option
        if len(available_embeddings) == 1:
            embedding_choice = available_embeddings[0]
            print(f"✓ Using {embedding_choice} for embeddings (only available provider)")
        else:
            print(f"\nAvailable embedding providers: {', '.join(available_embeddings)}")
            embedding_choice = prompt_choice("Select embedding provider", available_embeddings, embedding_default)

        # Vector DB choice
        available_vdbs = []
        if qdrant.available:
            available_vdbs.append("qdrant")
        if faiss.available:
            available_vdbs.append("faiss")

        if not available_vdbs:
            if RICH_AVAILABLE:
                console.print("\n[red]❌ No vector database available![/red]")
                console.print("Please either:")
                console.print("  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
                console.print("  2. Install FAISS: pip install faiss-cpu")
            else:
                print("\n❌ No vector database available!")
                print("Please either:")
                print("  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
                print("  2. Install FAISS: pip install faiss-cpu")
            raise typer.Exit(code=1)

        # Auto-select if only one option
        if len(available_vdbs) == 1:
            vector_db_choice = available_vdbs[0]
            print(f"✓ Using {vector_db_choice} for vector database (only available option)")
        else:
            print(f"\nAvailable vector databases: {', '.join(available_vdbs)}")
            vector_db_choice = prompt_choice("Select vector database", available_vdbs, vector_db_default)

        # Collection name
        if RICH_AVAILABLE:
            collection_name = Prompt.ask("\nCollection name", default="default")
        else:
            collection_name = input("\nCollection name [default]: ").strip() or "default"

        # Rerank option (if Cohere available)
        enable_rerank = False
        if api_keys["cohere"].available:
            print()
            enable_rerank = prompt_confirm("Enable reranking? (improves quality, uses Cohere API)", default=False)

    # ==========================================================================
    # Generate Config
    # ==========================================================================

    # Get Qdrant host if using Qdrant
    qdrant_host = "localhost"
    qdrant_port = 6333
    if vector_db_choice == "qdrant" and qdrant.available:
        # Parse from details
        if "At " in qdrant.details:
            try:
                addr = qdrant.details.split("At ")[1].split(" ")[0]
                qdrant_host, qdrant_port = addr.split(":")
                qdrant_port = int(qdrant_port)
            except:
                pass

    config = generate_config(
        llm_provider=llm_choice,
        embedding_provider=embedding_choice,
        vector_db=vector_db_choice,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection=collection_name,
        enable_rerank=enable_rerank if not non_interactive else False,
    )

    # ==========================================================================
    # Output
    # ==========================================================================

    print_header("Generated Configuration")

    if RICH_AVAILABLE:
        from rich.syntax import Syntax
        console.print(Syntax(config, "yaml", theme="monokai", line_numbers=False))
    else:
        print(config)

    if show_config:
        return

    # Save config
    print_header("Saving Configuration")

    # Determine save locations
    cwd = Path.cwd()
    fitz_dir = cwd / ".fitz"

    # Primary config locations that the code actually reads
    config_locations = [
        cwd / "fitz" / "engines" / "classic_rag" / "config" / "default.yaml",
        cwd / "fitz" / "pipeline" / "config" / "default.yaml",
    ]

    # Also save a copy to .fitz for reference
    fitz_config = fitz_dir / "config.yaml"

    if not non_interactive:
        # Check if any config exists
        existing = [p for p in config_locations if p.exists()]
        if existing or fitz_config.exists():
            if not prompt_confirm(f"Overwrite existing config files?", default=False):
                print("Aborted.")
                raise typer.Exit(code=0)

    # Create .fitz directory
    fitz_dir.mkdir(exist_ok=True)

    # Save to all locations that exist (don't create new directories)
    saved_to = []
    for config_path in config_locations:
        if config_path.parent.exists():
            config_path.write_text(config)
            saved_to.append(config_path)
            print(f"✓ Saved to {config_path.relative_to(cwd)}")

    # Always save to .fitz/config.yaml as backup/reference
    fitz_config.write_text(config)
    saved_to.append(fitz_config)
    print(f"✓ Saved to {fitz_config.relative_to(cwd)}")

    if not saved_to:
        print("⚠ No config directories found - only saved to .fitz/config.yaml")

    # ==========================================================================
    # Next Steps
    # ==========================================================================

    print_header("🎉 Setup Complete!")

    next_steps = f"""
Your configuration is ready! Next steps:

1. Ingest some documents:
   fitz-ingest run ./your_docs --collection {collection_name}

2. Query your knowledge base:
   fitz-pipeline query "Your question here"

3. (Optional) Test your setup:
   fitz doctor

Config saved to: .fitz/config.yaml
"""

    if RICH_AVAILABLE:
        console.print(Panel(next_steps, title="Next Steps", border_style="green"))
    else:
        print(next_steps)


# For running directly
if __name__ == "__main__":
    typer.run(command)