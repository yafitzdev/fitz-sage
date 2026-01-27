# tools/mutation_test.py
"""
Mutation testing runner for fitz-ai.

Usage:
    python tools/mutation_test.py           # Run mutation tests on all targets
    python tools/mutation_test.py --quick   # Run on core/ only (faster)
    python tools/mutation_test.py --show    # Show surviving mutants
    python tools/mutation_test.py --html    # Generate HTML report

Mutation testing is slow - expect 10-30 minutes for full run.
Recommended for nightly CI or pre-release validation only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_cmd(cmd: list[str], desc: str) -> int:
    """Run a command and return exit code."""
    print(f"\n{'=' * 60}")
    print(f"[MUTMUT] {desc}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main() -> int:
    args = set(sys.argv[1:])

    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0

    if "--show" in args:
        # Show surviving mutants
        return run_cmd(
            ["python", "-m", "mutmut", "results"],
            "Showing mutation test results",
        )

    if "--html" in args:
        # Generate HTML report
        run_cmd(
            ["python", "-m", "mutmut", "html"],
            "Generating HTML report",
        )
        print(f"\nReport generated: {PROJECT_ROOT / 'html' / 'index.html'}")
        return 0

    # Run mutation tests
    if "--quick" in args:
        # Quick mode: only core/
        cmd = [
            "python",
            "-m",
            "mutmut",
            "run",
            "--paths-to-mutate=fitz_ai/core/",
            "--paths-to-exclude=__init__.py,fitz_ai/core/paths.py,fitz_ai/core/constants.py",
        ]
        desc = "Running mutation tests (quick mode - core/ only)"
    else:
        # Full mode: use pyproject.toml config
        cmd = ["python", "-m", "mutmut", "run"]
        desc = "Running mutation tests (full mode)"

    exit_code = run_cmd(cmd, desc)

    # Show summary
    print("\n" + "=" * 60)
    print("MUTATION TESTING COMPLETE")
    print("=" * 60)

    # Show results
    run_cmd(["python", "-m", "mutmut", "results"], "Summary")

    if exit_code != 0:
        print("\nSome mutants survived. Run with --show for details.")
        print("Consider adding tests to kill surviving mutants.")
    else:
        print("\nAll mutants killed! Excellent test coverage.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
