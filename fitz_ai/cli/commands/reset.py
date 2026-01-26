# fitz_ai/cli/commands/reset.py
"""
Database reset command.

Usage:
    fitz reset              # Reset pgserver and wipe database
    fitz reset --force      # Skip confirmation prompt
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

import typer

from fitz_ai.cli.ui import ui
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

app = typer.Typer(help="Reset database and storage")


def _kill_postgres_processes() -> int:
    """Kill any running postgres/initdb processes. Returns count killed."""
    system = platform.system()
    killed = 0

    if system == "Windows":
        for proc in ["postgres.exe", "initdb.exe"]:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", proc],
                capture_output=True,
            )
            if result.returncode == 0:
                killed += 1
    else:
        # Unix-like systems
        for proc in ["postgres", "initdb"]:
            result = subprocess.run(
                ["pkill", "-9", "-f", proc],
                capture_output=True,
            )
            if result.returncode == 0:
                killed += 1

    return killed


def _wipe_pgdata() -> bool:
    """Delete pgdata directory. Returns True if deleted."""
    pgdata = FitzPaths.pgdata()
    if pgdata.exists():
        shutil.rmtree(pgdata, ignore_errors=True)
        return True
    return False


def _clear_pgserver_temp() -> None:
    """Clear any pgserver temp/state files."""
    import os
    import tempfile

    # Clear temp directory pgserver files
    temp_dir = tempfile.gettempdir()
    for item in ["pgserver", "pg_"]:
        for path in list(Path(temp_dir).glob(f"{item}*")):
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink()
            except Exception:
                pass

    # Clear app data (Windows)
    for env_var in ["LOCALAPPDATA", "APPDATA"]:
        app_data = os.environ.get(env_var)
        if app_data:
            pgserver_dir = Path(app_data) / "pgserver"
            if pgserver_dir.exists():
                shutil.rmtree(pgserver_dir, ignore_errors=True)


@app.callback(invoke_without_command=True)
def reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Reset pgserver database completely.

    This command:
    1. Kills any running postgres processes
    2. Deletes the pgdata directory
    3. Clears pgserver temp/state files

    Use this when pgserver gets stuck or corrupted.
    """
    if not force:
        confirm = typer.confirm(
            "This will kill postgres processes and wipe the database. Continue?"
        )
        if not confirm:
            ui.warning("Aborted.")
            raise typer.Exit(0)

    ui.info("Resetting pgserver...")

    # Step 1: Kill processes
    killed = _kill_postgres_processes()
    if killed:
        ui.success(f"Killed {killed} postgres process(es)")
    else:
        ui.info("No postgres processes found")

    # Step 2: Wipe pgdata
    if _wipe_pgdata():
        ui.success(f"Deleted {FitzPaths.pgdata()}")
    else:
        ui.info("pgdata directory not found")

    # Step 3: Clear temp files
    _clear_pgserver_temp()
    ui.success("Cleared pgserver temp files")

    ui.success("Reset complete. Database will reinitialize on next use.")
