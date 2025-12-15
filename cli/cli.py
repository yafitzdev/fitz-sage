"""
Main fitz CLI for setup and utility commands.

This is the top-level `fitz` command that provides:
- setup-local: Set up local LLM (Ollama) for offline testing
- Other utility commands (future)
"""
import subprocess
import sys

import typer

app = typer.Typer(
    help="Fitz CLI - Setup and utility commands",
    no_args_is_help=True,
)


def check_ollama_installed() -> bool:
    """Check if Ollama is installed and accessible."""
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


def pull_model() -> bool:
    """Pull the llama3.2:1b model using Ollama."""
    try:
        typer.echo("Pulling model llama3.2:1b...")
        typer.echo("(This may take a few minutes)")
        typer.echo()

        # Run ollama pull and show output
        result = subprocess.run(
            ["ollama", "pull", "llama3.2:1b"],
            check=False,
        )

        return result.returncode == 0
    except Exception as e:
        typer.echo(f"Error pulling model: {e}")
        return False


def verify_local_llm() -> bool:
    """Verify the local LLM works by testing it."""
    try:
        # Import here to avoid issues if backends aren't available
        from fitz.backends.local_llm.runtime import LocalLLMRuntime, LocalLLMRuntimeConfig

        typer.echo()
        typer.echo("Verifying local LLM...")

        # Try to initialize and use the runtime
        runtime_cfg = LocalLLMRuntimeConfig(model="llama3.2:1b")
        runtime = LocalLLMRuntime(runtime_cfg)

        # This will raise an exception if Ollama isn't running or model isn't available
        adapter = runtime.llama()

        typer.echo("✓ Local LLM verified")
        return True

    except Exception as e:
        typer.echo(f"✗ Verification failed: {e}")
        return False


@app.command("setup-local")
def setup_local() -> None:
    """
    Set up local LLM (Ollama) for offline testing.

    This command guides you through setting up Ollama as a local LLM fallback.
    The local LLM is optional and only required for offline testing.

    Steps:
    1. Checks if Ollama is installed
    2. Pulls the llama3.2:1b model
    3. Verifies the setup works

    Examples:
        # Run setup
        fitz setup-local

        # After setup, test it works
        fitz test
    """
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("FITZ LOCAL LLM SETUP")
    typer.echo("=" * 60)
    typer.echo()

    # Step 1: Check if Ollama is installed
    if not check_ollama_installed():
        typer.echo("Ollama is not installed.")
        typer.echo()
        typer.echo("Fitz uses Ollama for its local LLM fallback.")
        typer.echo("This is optional and only required for offline testing.")
        typer.echo()
        typer.echo("Please install Ollama manually:")
        typer.echo("  https://ollama.com")
        typer.echo()
        typer.echo("After installation, rerun:")
        typer.echo("  fitz setup-local")
        typer.echo()
        raise typer.Exit(code=1)

    typer.echo("✓ Ollama detected")
    typer.echo()

    # Step 2: Pull the model
    if not pull_model():
        typer.echo()
        typer.echo("✗ Failed to pull model")
        typer.echo()
        typer.echo("Please ensure Ollama is running and try again.")
        raise typer.Exit(code=1)

    typer.echo()
    typer.echo("✓ Model downloaded")

    # Step 3: Verify
    if not verify_local_llm():
        typer.echo()
        typer.echo("Setup completed but verification failed.")
        typer.echo("The model is downloaded, but Ollama may not be running.")
        typer.echo()
        typer.echo("Try:")
        typer.echo("  1. Start Ollama (it usually starts automatically)")
        typer.echo("  2. Run: fitz setup-local")
        raise typer.Exit(code=1)

    # Success!
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("✓ Local LLM setup complete")
    typer.echo("=" * 60)
    typer.echo()
    typer.echo("You can now run:")
    typer.echo("  fitz test")
    typer.echo()


@app.command("test")
def test() -> None:
    """
    Test local LLM setup.

    Runs a quick test to verify that:
    - Ollama is installed
    - The model is available
    - Local LLM can generate responses

    Examples:
        # Test local setup
        fitz test
    """
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("TESTING LOCAL LLM")
    typer.echo("=" * 60)
    typer.echo()

    # Check Ollama
    typer.echo("[1/3] Checking Ollama installation...")
    if not check_ollama_installed():
        typer.echo("  ✗ Ollama not found")
        typer.echo()
        typer.echo("Run: fitz setup-local")
        raise typer.Exit(code=1)
    typer.echo("  ✓ Ollama detected")

    # Check model
    typer.echo()
    typer.echo("[2/3] Checking model availability...")
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "llama3.2:1b" not in result.stdout:
            typer.echo("  ✗ Model llama3.2:1b not found")
            typer.echo()
            typer.echo("Run: fitz setup-local")
            raise typer.Exit(code=1)
        typer.echo("  ✓ Model llama3.2:1b available")
    except Exception as e:
        typer.echo(f"  ✗ Error checking model: {e}")
        raise typer.Exit(code=1)

    # Test generation
    typer.echo()
    typer.echo("[3/3] Testing text generation...")
    try:
        from fitz.backends.local_llm.runtime import LocalLLMRuntime, LocalLLMRuntimeConfig

        runtime_cfg = LocalLLMRuntimeConfig(model="llama3.2:1b")
        runtime = LocalLLMRuntime(runtime_cfg)
        adapter = runtime.llama()

        # Test a simple chat
        response = adapter.chat([
            {"role": "user", "content": "Say 'hello' in one word."}
        ])

        if response:
            typer.echo("  ✓ Generation successful")
            typer.echo(f"  Response preview: {response[:50]}...")
        else:
            typer.echo("  ⚠ Generation returned empty response")

    except Exception as e:
        typer.echo(f"  ✗ Generation failed: {e}")
        typer.echo()
        typer.echo("This might mean:")
        typer.echo("  • Ollama service is not running")
        typer.echo("  • Model is not properly installed")
        typer.echo()
        typer.echo("Try: fitz setup-local")
        raise typer.Exit(code=1)

    # Success!
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("✓ ALL TESTS PASSED")
    typer.echo("=" * 60)
    typer.echo()
    typer.echo("Your local LLM is working correctly!")
    typer.echo()


if __name__ == "__main__":
    app()