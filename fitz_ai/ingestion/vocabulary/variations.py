# fitz_ai/ingestion/vocabulary/variations.py
"""
Variation generation for keywords.

Generates probable variations of detected keywords to handle
different formatting conventions (spaces, underscores, hyphens).
"""

from __future__ import annotations

import re


def generate_variations(detected: str, category: str) -> list[str]:
    """
    Generate probable variations of a detected keyword.

    This handles common formatting differences:
    - Separator variations (space, underscore, hyphen, none)
    - Case variations (lower, upper, original)
    - Category-specific expansions (TC-1001 → testcase 1001)

    Args:
        detected: The originally detected keyword string
        category: The category of the keyword (testcase, ticket, version, etc.)

    Returns:
        List of unique variations including the original
    """
    variations: set[str] = {detected}

    # Normalize: replace separators with spaces
    normalized = re.sub(r"[\s_\-]+", " ", detected).strip()
    variations.add(normalized)

    # Separator variations
    variations.add(normalized.replace(" ", "_"))  # underscore
    variations.add(normalized.replace(" ", "-"))  # hyphen
    variations.add(normalized.replace(" ", ""))  # no separator

    # Case variations
    variations.add(detected.lower())
    variations.add(detected.upper())
    variations.add(normalized.lower())
    variations.add(normalized.upper())

    # Category-specific expansions
    category_variations = _generate_category_variations(detected, category)
    variations.update(category_variations)

    # Remove empty strings and duplicates
    return sorted([v for v in variations if v], key=str.lower)


def _generate_category_variations(detected: str, category: str) -> set[str]:
    """Generate category-specific variations."""
    variations: set[str] = set()

    if category == "testcase":
        # TC-1001 → testcase 1001, test case 1001, test_1001
        match = re.match(r"TC[_\-\s]?(\d+)", detected, re.IGNORECASE)
        if match:
            num = match.group(1)
            variations.update(
                [
                    f"testcase {num}",
                    f"testcase_{num}",
                    f"testcase-{num}",
                    f"test case {num}",
                    f"test_case_{num}",
                    f"test-case-{num}",
                    f"test {num}",
                    f"test_{num}",
                    f"TC{num}",
                    f"TC-{num}",
                    f"TC_{num}",
                    f"tc{num}",
                    f"tc-{num}",
                    f"tc_{num}",
                ]
            )

    elif category == "ticket":
        # JIRA-1234 → jira 1234, JIRA 1234
        match = re.match(r"([A-Z]+)[_\-\s]?(\d+)", detected, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            num = match.group(2)
            variations.update(
                [
                    f"{prefix} {num}",
                    f"{prefix.lower()} {num}",
                    f"{prefix.upper()}{num}",
                    f"{prefix.lower()}{num}",
                    f"{prefix.upper()}-{num}",
                    f"{prefix.lower()}-{num}",
                    f"{prefix.upper()}_{num}",
                    f"{prefix.lower()}_{num}",
                ]
            )

    elif category == "pull_request":
        # PR-123 → pull request 123, PR#123, PR 123
        match = re.match(r"PR[_\-\s#]?(\d+)", detected, re.IGNORECASE)
        if match:
            num = match.group(1)
            variations.update(
                [
                    f"PR {num}",
                    f"PR-{num}",
                    f"PR#{num}",
                    f"PR{num}",
                    f"pr {num}",
                    f"pr-{num}",
                    f"pr#{num}",
                    f"pr{num}",
                    f"pull request {num}",
                    f"pull request #{num}",
                    f"pull-request-{num}",
                ]
            )

    elif category == "version":
        # v2.0.1 → version 2.0.1, 2.0.1
        match = re.match(r"v?(\d+\.\d+(?:\.\d+)?(?:-\w+)?)", detected, re.IGNORECASE)
        if match:
            ver = match.group(1)
            variations.update(
                [
                    f"v{ver}",
                    f"V{ver}",
                    ver,
                    f"version {ver}",
                    f"Version {ver}",
                    f"release {ver}",
                    f"Release {ver}",
                ]
            )

    elif category == "person":
        # John Smith → J. Smith, jsmith
        parts = detected.split()
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            variations.update(
                [
                    f"{first[0]}. {last}",  # J. Smith
                    f"{first[0].lower()}. {last.lower()}",  # j. smith
                    f"{first[0].lower()}{last.lower()}",  # jsmith
                    f"{first.lower()}.{last.lower()}",  # john.smith
                    f"{first.lower()}_{last.lower()}",  # john_smith
                ]
            )

    return variations


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for matching by standardizing separators and case.

    Args:
        text: Text to normalize

    Returns:
        Normalized text (lowercase, separators → spaces)
    """
    return re.sub(r"[\s_\-]+", " ", text.lower()).strip()
