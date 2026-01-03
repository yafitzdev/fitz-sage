# tools/cli_map/__main__.py
"""
CLI Map - extract and display CLI structure from the Typer app.

Usage:
    python -m tools.cli_map            # Full demo with parallel queries (~2 min)
    python -m tools.cli_map --minimal  # Skip code corpus (~1 min)
    python -m tools.cli_map --help-only  # Just --help text for all commands
    python -m tools.cli_map --compact  # One-line summary per command
    python -m tools.cli_map --full     # Full details with types and defaults
    python -m tools.cli_map --json     # JSON output
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Ensure repo root is in path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


@dataclass
class ParamInfo:
    """Information about a command parameter (argument or option)."""

    name: str
    kind: str  # "argument" or "option"
    type_hint: str
    required: bool
    default: Optional[str]
    help: str
    flags: list[str] = field(default_factory=list)  # e.g., ["--verbose", "-v"]


@dataclass
class CommandInfo:
    """Information about a CLI command."""

    name: str
    help: str
    arguments: list[ParamInfo] = field(default_factory=list)
    options: list[ParamInfo] = field(default_factory=list)


@dataclass
class CLIMap:
    """Complete CLI structure."""

    app_name: str
    app_help: str
    commands: list[CommandInfo] = field(default_factory=list)
    subapps: dict[str, "CLIMap"] = field(default_factory=dict)


def extract_param_info(param: Any) -> ParamInfo:
    """Extract parameter info from a Click parameter."""
    import click

    # Determine kind
    if isinstance(param, click.Argument):
        kind = "argument"
        flags = []
    else:
        kind = "option"
        flags = list(param.opts) if hasattr(param, "opts") else []

    # Get type hint
    type_hint = "str"
    if param.type:
        type_name = getattr(param.type, "name", None)
        if type_name:
            type_hint = type_name
        elif hasattr(param.type, "__class__"):
            type_hint = param.type.__class__.__name__

    # Get default
    default = None
    if param.default is not None and param.default != ():
        default = repr(param.default)

    # Get help
    help_text = getattr(param, "help", "") or ""

    return ParamInfo(
        name=param.name,
        kind=kind,
        type_hint=type_hint,
        required=param.required,
        default=default,
        help=help_text,
        flags=flags,
    )


def extract_command_info(name: str, cmd: Any) -> CommandInfo:
    """Extract command info from a Click/Typer command."""
    help_text = cmd.help or cmd.callback.__doc__ or ""
    help_text = help_text.strip().split("\n")[0]  # First line only

    arguments = []
    options = []

    for param in cmd.params:
        info = extract_param_info(param)
        if info.kind == "argument":
            arguments.append(info)
        else:
            # Skip the --help option
            if info.name != "help":
                options.append(info)

    return CommandInfo(
        name=name,
        help=help_text,
        arguments=arguments,
        options=options,
    )


def extract_cli_map(app: Any) -> CLIMap:
    """Extract the full CLI map from a Typer app."""
    import click
    import typer

    # Get underlying click group from Typer
    click_app = typer.main.get_command(app)

    app_name = getattr(click_app, "name", "fitz") or "fitz"
    app_help = getattr(click_app, "help", "") or ""

    cli_map = CLIMap(app_name=app_name, app_help=app_help)

    # Extract commands
    if hasattr(click_app, "commands"):
        for cmd_name, cmd in sorted(click_app.commands.items()):
            if isinstance(cmd, click.Group):
                # Subapp (e.g., benchmark) - recurse
                sub_map = CLIMap(app_name=cmd_name, app_help=cmd.help or "")
                if hasattr(cmd, "commands"):
                    for sub_name, sub_cmd in sorted(cmd.commands.items()):
                        sub_map.commands.append(extract_command_info(sub_name, sub_cmd))
                cli_map.subapps[cmd_name] = sub_map
            else:
                cli_map.commands.append(extract_command_info(cmd_name, cmd))

    return cli_map


def render_compact(cli_map: CLIMap) -> str:
    """Render CLI map in compact format - one line per command."""
    lines = []

    # Stats
    total_commands = len(cli_map.commands)
    total_args = sum(len(c.arguments) for c in cli_map.commands)
    total_opts = sum(len(c.options) for c in cli_map.commands)
    total_subcommands = sum(len(s.commands) for s in cli_map.subapps.values())

    # Header
    lines.append("=" * 72)
    lines.append(f"  CLI MAP: {cli_map.app_name}")
    lines.append(
        f"  {total_commands} commands | {total_args} args | {total_opts} options | {len(cli_map.subapps)} subapps ({total_subcommands} subcommands)"
    )
    lines.append("=" * 72)
    lines.append("")

    # Commands
    for cmd in cli_map.commands:
        # Build signature
        sig_parts = [f"fitz {cmd.name}"]

        for arg in cmd.arguments:
            if arg.required:
                sig_parts.append(f"<{arg.name}>")
            else:
                sig_parts.append(f"[{arg.name}]")

        for opt in cmd.options:
            flag = opt.flags[0] if opt.flags else f"--{opt.name}"
            if opt.type_hint == "BOOL" or opt.type_hint == "boolean":
                sig_parts.append(f"[{flag}]")
            elif opt.required:
                sig_parts.append(f"{flag}=<{opt.type_hint}>")
            else:
                sig_parts.append(f"[{flag}]")

        sig = " ".join(sig_parts)
        lines.append(f"  {sig}")
        lines.append(f"    └─ {cmd.help}")
        lines.append("")

    # Subapps
    for subapp_name, subapp in sorted(cli_map.subapps.items()):
        lines.append(f"  fitz {subapp_name} ...")
        lines.append(f"    └─ {subapp.app_help or '(subcommand group)'}")
        for cmd in subapp.commands:
            sig_parts = [f"      fitz {subapp_name} {cmd.name}"]
            for arg in cmd.arguments:
                sig_parts.append(f"[{arg.name}]" if not arg.required else f"<{arg.name}>")
            sig = " ".join(sig_parts)
            lines.append(sig)
        lines.append("")

    lines.append("-" * 72)

    return "\n".join(lines)


def render_full(cli_map: CLIMap) -> str:
    """Render CLI map with full details - types, defaults, help text."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append(f"CLI MAP: {cli_map.app_name}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(cli_map.app_help)
    lines.append("")
    lines.append("-" * 80)

    # Commands
    for cmd in cli_map.commands:
        lines.append("")
        lines.append(f"┌── fitz {cmd.name}")
        lines.append(f"│   {cmd.help}")
        lines.append("│")

        if cmd.arguments:
            lines.append("│   Arguments:")
            for arg in cmd.arguments:
                req = "required" if arg.required else "optional"
                default = f" = {arg.default}" if arg.default else ""
                lines.append(f"│     {arg.name} : {arg.type_hint} ({req}{default})")
                if arg.help:
                    lines.append(f"│       └─ {arg.help}")

        if cmd.options:
            lines.append("│   Options:")
            for opt in cmd.options:
                flags = ", ".join(opt.flags) if opt.flags else f"--{opt.name}"
                req = "required" if opt.required else "optional"
                default = f" = {opt.default}" if opt.default else ""
                lines.append(f"│     {flags} : {opt.type_hint} ({req}{default})")
                if opt.help:
                    lines.append(f"│       └─ {opt.help}")

        lines.append("└" + "─" * 40)

    # Subapps
    for subapp_name, subapp in sorted(cli_map.subapps.items()):
        lines.append("")
        lines.append(f"┌── fitz {subapp_name} (subcommand group)")
        lines.append(f"│   {subapp.app_help}")
        lines.append("│")

        for cmd in subapp.commands:
            lines.append(f"│   ├── {cmd.name}")
            lines.append(f"│   │   {cmd.help}")

            if cmd.arguments:
                for arg in cmd.arguments:
                    req = "required" if arg.required else "optional"
                    lines.append(f"│   │     {arg.name} : {arg.type_hint} ({req})")

            if cmd.options:
                for opt in cmd.options:
                    flags = ", ".join(opt.flags) if opt.flags else f"--{opt.name}"
                    lines.append(f"│   │     {flags} : {opt.type_hint}")

        lines.append("└" + "─" * 40)

    return "\n".join(lines)


def render_json(cli_map: CLIMap) -> str:
    """Render CLI map as JSON."""

    def to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for k, v in asdict(obj).items():
                result[k] = v
            return result
        return obj

    return json.dumps(asdict(cli_map), indent=2)


def render_live_output(minimal: bool = False) -> str:
    """
    Run actual CLI commands against test data and capture real output.

    Creates temp test data, runs commands, captures output, cleans up.
    Queries run in parallel for speed (~2x faster).

    Args:
        minimal: If True, skip code corpus entirely (quickstart + docs only)
    """
    import os
    import shutil
    import subprocess
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    lines = []

    # Find fitz executable
    fitz_exe = shutil.which("fitz")
    if not fitz_exe:
        venv_dir = Path(sys.executable).parent
        fitz_exe = venv_dir / "fitz.exe"
        if not fitz_exe.exists():
            fitz_exe = venv_dir / "fitz"
        fitz_exe = str(fitz_exe)

    def run_cmd(*args: str, timeout: int = 60, stdin_input: str = None) -> str:
        """Run fitz command and capture output.

        Args:
            *args: Command arguments
            timeout: Timeout in seconds
            stdin_input: Input to send to stdin (simulates user typing + Enter)
        """
        try:
            env = {**os.environ, "NO_COLOR": "1", "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run(
                [fitz_exe, *args],
                capture_output=True,
                timeout=timeout,
                env=env,
                encoding="utf-8",
                errors="replace",
                input=stdin_input,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            output = stdout + stderr
            return output.strip() if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return "(timed out)"
        except Exception as e:
            return f"ERROR: {e}"

    def section(title: str, output: str) -> None:
        """Add a section to the output."""
        lines.append("=" * 72)
        lines.append(f"  {title}")
        lines.append("=" * 72)
        lines.append(output)
        lines.append("")

    # Create temp directory with test documents
    temp_dir = tempfile.mkdtemp(prefix="fitz_cli_map_")
    test_collection = "cli_map_test"

    try:
        # Create test documents
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()

        # =====================================================================
        # Header
        # =====================================================================
        lines.append("=" * 72)
        lines.append("  CLI MAP - Live Output Demo")
        lines.append("=" * 72)
        lines.append("")

        # =====================================================================
        # Engine Management
        # =====================================================================
        section(
            "fitz engine --list",
            run_cmd("engine", "--list"),
        )

        # Create test documents (.md, .txt) - needed for all runs
        (docs_dir / "about.md").write_text(
            "# About Fitz\n\n"
            "Fitz is a local-first RAG framework for building knowledge engines.\n\n"
            "## Features\n"
            "- Zero-config quickstart\n"
            "- Epistemic honesty (says 'I don't know')\n"
            "- Full provenance tracking\n",
            encoding="utf-8",
        )

        (docs_dir / "installation.txt").write_text(
            "Installation Instructions\n"
            "=========================\n\n"
            "To install Fitz, run:\n"
            "  pip install fitz-ai\n\n"
            "Requirements:\n"
            "- Python 3.10 or higher\n"
            "- 8GB RAM recommended\n",
            encoding="utf-8",
        )

        (docs_dir / "changelog.md").write_text(
            "# Changelog\n\n"
            "## v0.4.0 (2024-01-15)\n"
            "- Added hierarchical summaries\n"
            "- Improved citation accuracy\n\n"
            "## v0.3.0 (2024-01-01)\n"
            "- Initial public release\n",
            encoding="utf-8",
        )

        # Create a simple PDF-like content (as .txt since we can't create real PDFs easily)
        # Note: Real PDF testing would need a PDF file
        (docs_dir / "license.txt").write_text(
            "MIT License\n\n"
            "Copyright (c) 2024 Fitz AI\n\n"
            "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
            "of this software and associated documentation files.\n",
            encoding="utf-8",
        )

        # =====================================================================
        # RUN 0: Quickstart (Zero-Config)
        # =====================================================================
        lines.append("=" * 72)
        lines.append("  RUN 0: Quickstart (Zero-Config)")
        lines.append(f"  Test data: {docs_dir}")
        lines.append("  One command to ingest + query!")
        lines.append("=" * 72)
        lines.append("")

        # fitz quickstart - the simplest entry point
        section(
            f'fitz quickstart {docs_dir} "What is Fitz?"',
            run_cmd("quickstart", str(docs_dir), "What is Fitz?", stdin_input="\n", timeout=180),
        )

        # Clean up quickstart collection
        from fitz_ai.cli.context import CLIContext

        ctx = CLIContext.load()
        client = ctx.get_vector_db_client()
        qs_collections = sorted(client.list_collections())
        if "quickstart" in qs_collections:
            qs_index = qs_collections.index("quickstart") + 1
            section(
                "fitz collections  [interactive: delete quickstart]",
                run_cmd("collections", stdin_input=f"\n{qs_index}\n2\ny\n4\n"),
            )

        # =====================================================================
        # RUN 1: Full Interactive Flow (Document Corpus)
        # =====================================================================
        lines.append("=" * 72)
        lines.append("  RUN 1: Full Interactive Flow (Document Corpus)")
        lines.append(f"  Test data: {docs_dir}")
        lines.append(f"  Collection: {test_collection}")
        lines.append("=" * 72)
        lines.append("")

        # 1-2. fitz doctor & config (run in parallel - no dependencies)
        with ThreadPoolExecutor(max_workers=2) as executor:
            doctor_future = executor.submit(run_cmd, "doctor")
            config_future = executor.submit(run_cmd, "config")
            doctor_output = doctor_future.result()
            config_output = config_future.result()

        section("fitz doctor", doctor_output)
        section("fitz config", config_output)

        # 3. fitz ingest documents
        # Reload client to get fresh state from disk (subprocess may have modified it)
        ctx = CLIContext.load()
        client = ctx.get_vector_db_client()
        existing_collections = client.list_collections()
        # If no collections, menu is skipped → just send collection name
        # If collections exist, menu shows → send "0" to select "Create new", then name
        # Note: Engine selection no longer prompts (uses default), so no leading \n needed
        if existing_collections:
            ingest_stdin = f"0\n{test_collection}\n"
        else:
            ingest_stdin = f"{test_collection}\n"

        section(
            f"fitz ingest {docs_dir}  [interactive: create '{test_collection}']",
            run_cmd("ingest", str(docs_dir), stdin_input=ingest_stdin),
        )

        # 4. fitz collections (show list and exit)
        # Menu: [1] vector_db selection → Enter, [1] collection, [2] Exit → send "2"
        section(
            "fitz collections  [interactive: Enter, 2 to exit]",
            run_cmd("collections", stdin_input="\n2\n"),
        )

        # 5. fitz query - document questions (run in parallel!)
        # Note: Engine selection no longer prompts, so no stdin needed
        doc_queries = [
            ("What is Fitz?", f'fitz query "What is Fitz?" -c {test_collection}'),
            ("How do I install Fitz?", f'fitz query "How do I install Fitz?" -c {test_collection}'),
            (
                "What changed in version 0.4?",
                f'fitz query "What changed in version 0.4?" -c {test_collection}',
            ),
        ]

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                (
                    q[1],
                    executor.submit(run_cmd, "query", q[0], "-c", test_collection, timeout=120),
                )
                for q in doc_queries
            ]
            for title, future in futures:
                section(f"{title}  [parallel]", future.result())

        # 6. fitz collections - delete collection (interactive)
        # Reload client to get fresh state from disk
        ctx = CLIContext.load()
        client = ctx.get_vector_db_client()
        collections = sorted(client.list_collections())
        coll_index = collections.index(test_collection) + 1 if test_collection in collections else 1

        section(
            f"fitz collections  [interactive: delete {test_collection}]",
            run_cmd("collections", stdin_input=f"\n{coll_index}\n2\ny\n4\n"),
        )

        # =====================================================================
        # RUN 2: Code corpus (.py files - contract_map)
        # =====================================================================
        if not minimal:
            code_collection = "cli_map_code_test"
            contract_map_dir = Path(__file__).parent.parent / "contract_map"

            lines.append("=" * 72)
            lines.append("  RUN 2: Code Corpus (.py files)")
            lines.append(f"  Source: {contract_map_dir}")
            lines.append(f"  Collection: {code_collection}")
            lines.append("=" * 72)
            lines.append("")

            # 6. fitz ingest code
            # Reload client to get fresh state from disk (subprocess may have modified it)
            ctx = CLIContext.load()
            client = ctx.get_vector_db_client()
            existing_collections = client.list_collections()
            # Note: Engine selection no longer prompts (uses default), so no leading \n needed
            if existing_collections:
                ingest_stdin = f"0\n{code_collection}\n"
            else:
                ingest_stdin = f"{code_collection}\n"

            section(
                f"fitz ingest {contract_map_dir}  [interactive: create '{code_collection}']",
                run_cmd("ingest", str(contract_map_dir), stdin_input=ingest_stdin),
            )

            # 7. fitz query - code questions (run in parallel!)
            # Note: Engine selection no longer prompts, so no stdin needed
            code_queries = [
                (
                    "What does the contract_map tool do?",
                    f'fitz query "What does the contract_map tool do?" -c {code_collection}',
                ),
                (
                    "How are architecture violations detected?",
                    f'fitz query "How are architecture violations detected?" -c {code_collection}',
                ),
                (
                    "What functions are in the imports module?",
                    f'fitz query "What functions are in the imports module?" -c {code_collection}',
                ),
            ]

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    (
                        q[1],
                        executor.submit(
                            run_cmd,
                            "query",
                            q[0],
                            "-c",
                            code_collection,
                            timeout=120,
                        ),
                    )
                    for q in code_queries
                ]
                for title, future in futures:
                    section(f"{title}  [parallel]", future.result())

            # 8. fitz collections - delete code collection (interactive)
            # Reload client to get fresh state from disk
            ctx = CLIContext.load()
            client = ctx.get_vector_db_client()
            collections = sorted(client.list_collections())
            code_coll_index = (
                collections.index(code_collection) + 1 if code_collection in collections else 1
            )

            section(
                f"fitz collections  [interactive: delete {code_collection}]",
                run_cmd("collections", stdin_input=f"\n{code_coll_index}\n2\ny\n4\n"),
            )

        # Mark that we cleaned up via CLI
        test_collection = None  # Don't try to delete again in finally block

    finally:
        # Cleanup: delete temp files only (collections deleted via CLI above)
        lines.append("=" * 72)
        lines.append("  Final Cleanup")
        lines.append("=" * 72)

        # Delete temp directory
        try:
            shutil.rmtree(temp_dir)
            lines.append(f"Deleted temp dir: {temp_dir}")
        except Exception as e:
            lines.append(f"Could not delete temp dir: {e}")

        lines.append("")

    return "\n".join(lines)


def render_help_output(cli_map: CLIMap) -> str:
    """Render actual --help output for all commands."""
    import shutil
    import subprocess

    lines = []

    # Find fitz executable
    fitz_exe = shutil.which("fitz")
    if not fitz_exe:
        # Try in the same venv as current python
        venv_dir = Path(sys.executable).parent
        fitz_exe = venv_dir / "fitz"
        if not fitz_exe.exists():
            fitz_exe = venv_dir / "fitz.exe"
        fitz_exe = str(fitz_exe)

    def run_help(*args: str) -> str:
        """Run fitz with args and capture output."""
        import os

        try:
            # Disable rich formatting for cleaner output
            env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run(
                [fitz_exe, *args, "--help"],
                capture_output=True,
                timeout=10,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            return stdout.strip() or stderr.strip()
        except Exception as e:
            return f"ERROR: {e}"

    # Main app help
    lines.append("=" * 72)
    lines.append("  fitz --help")
    lines.append("=" * 72)
    lines.append(run_help())
    lines.append("")

    # Each command's help
    for cmd in cli_map.commands:
        lines.append("=" * 72)
        lines.append(f"  fitz {cmd.name} --help")
        lines.append("=" * 72)
        lines.append(run_help(cmd.name))
        lines.append("")

    # Subapps
    for subapp_name, subapp in sorted(cli_map.subapps.items()):
        lines.append("=" * 72)
        lines.append(f"  fitz {subapp_name} --help")
        lines.append("=" * 72)
        lines.append(run_help(subapp_name))
        lines.append("")

        # Subcommands
        for cmd in subapp.commands:
            lines.append("-" * 72)
            lines.append(f"  fitz {subapp_name} {cmd.name} --help")
            lines.append("-" * 72)
            lines.append(run_help(subapp_name, cmd.name))
            lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Display CLI structure at a glance.")
    parser.add_argument(
        "--help-only",
        action="store_true",
        help="Show only --help text (no live execution)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Show compact overview (one line per command)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full details (types, defaults, help)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Minimal mode: quickstart + docs only, skip code corpus (~1 min)",
    )

    args = parser.parse_args(argv)

    # Ensure UTF-8 output on Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # Import the CLI app
    try:
        from fitz_ai.cli import app
    except ImportError as e:
        print(f"Error: Could not import CLI app: {e}", file=sys.stderr)
        return 1

    # Extract CLI structure
    cli_map = extract_cli_map(app)

    # Render output (live execution is default)
    if args.json:
        print(render_json(cli_map))
    elif args.compact:
        print(render_compact(cli_map))
    elif args.full:
        print(render_full(cli_map))
    elif args.help_only:
        print(render_help_output(cli_map))
    else:
        # Default: run actual commands with test data (queries run in parallel)
        print(render_live_output(minimal=args.minimal))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
