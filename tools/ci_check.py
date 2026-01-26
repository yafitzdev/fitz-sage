# tools/ci_check.py
"""
CI check script - run this before pushing to catch all issues.

Usage:
    python tools/ci_check.py          # Format + lint
    python tools/ci_check.py --test   # Format + lint + tier1 tests
"""

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], desc: str, fix: bool = False) -> bool:
    """Run a command and return success status."""
    prefix = "[FIX]" if fix else "[CHK]"
    print(f"\n{prefix} {desc}...")
    print(f"     {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        print("     PASSED")
        return True
    else:
        print("     FAILED")
        return False


def main():
    run_tests = "--test" in sys.argv

    print("=" * 60)
    print("CI Check Script")
    print("=" * 60)

    steps = [
        # Formatters (auto-fix)
        (["python", "-m", "black", "."], "Black (formatting)", True),
        (["python", "-m", "isort", "."], "isort (import sorting)", True),
        (["python", "-m", "ruff", "check", "--fix", "."], "Ruff --fix (auto-fix lints)", True),
        # Checkers (verify)
        (["python", "-m", "black", "--check", "."], "Black (verify)", False),
        (["python", "-m", "isort", "--check-only", "."], "isort (verify)", False),
        (["python", "-m", "ruff", "check", "."], "Ruff (verify)", False),
    ]

    if run_tests:
        steps.append((["python", "-m", "pytest", "-m", "tier1", "-q"], "Pytest tier1", False))

    failed = []
    for cmd, desc, fix in steps:
        if not run(cmd, desc, fix):
            failed.append(desc)

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        print("Fix the issues above before pushing.")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        print("Safe to push!")
        sys.exit(0)


if __name__ == "__main__":
    main()
