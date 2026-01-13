# fitz_ai/cli/commands/keywords.py
"""
Keyword vocabulary management commands.

Commands:
    fitz keywords list            - List all detected keywords
    fitz keywords add             - Add a custom keyword
    fitz keywords edit            - Edit keyword variations
    fitz keywords remove          - Remove a keyword
    fitz keywords suggest         - Suggest keywords from corpus
    fitz keywords detect          - Re-detect keywords from collection
"""

from __future__ import annotations

from typing import Optional

import typer

from fitz_ai.cli.ui import RICH, console, ui
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="keywords",
    help="Manage keyword vocabulary for exact matching.",
    no_args_is_help=True,
)


def _get_collection(collection: Optional[str]) -> str:
    """Get collection name, using default from context if not specified."""
    if collection is None:
        from fitz_ai.cli.context import CLIContext

        ctx = CLIContext.load()
        return ctx.retrieval_collection
    return collection


@app.command("list")
def list_keywords(
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses default if not specified.",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        help="Filter by category (testcase, ticket, version, etc.)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show all variations for each keyword.",
    ),
) -> None:
    """List all keywords in the vocabulary."""
    from fitz_ai.retrieval.vocabulary import VocabularyStore

    collection = _get_collection(collection)
    store = VocabularyStore(collection=collection)

    if not store.exists():
        ui.warning(f"No vocabulary file found for collection '{collection}'.")
        ui.info(
            "Run 'fitz ingest' to auto-detect keywords, or 'fitz keywords add' to add manually."
        )
        return

    keywords, metadata = store.load_with_metadata()

    if not keywords:
        ui.info("Vocabulary is empty.")
        return

    # Filter by category if specified
    if category:
        keywords = [kw for kw in keywords if kw.category == category]
        if not keywords:
            ui.warning(f"No keywords found in category '{category}'.")
            return

    # Show metadata
    ui.info(f"Collection: {collection}")
    if metadata:
        ui.info(
            f"Vocabulary: {metadata.auto_detected} auto-detected, {metadata.user_modified} user-defined"
        )
    print()

    # Group by category
    by_category: dict[str, list] = {}
    for kw in keywords:
        if kw.category not in by_category:
            by_category[kw.category] = []
        by_category[kw.category].append(kw)

    if RICH:
        from rich.table import Table

        for cat, cat_keywords in sorted(by_category.items()):
            table = Table(
                title=f"{cat.upper()} ({len(cat_keywords)})",
                show_header=True,
                header_style="bold",
            )
            table.add_column("ID", style="cyan")
            table.add_column("Occurrences", justify="right")
            if verbose:
                table.add_column("Variations")

            for kw in cat_keywords:
                if verbose:
                    variations = ", ".join(kw.match[:5])
                    if len(kw.match) > 5:
                        variations += f" (+{len(kw.match) - 5} more)"
                    table.add_row(kw.id, str(kw.occurrences), variations)
                else:
                    table.add_row(kw.id, str(kw.occurrences))

            console.print(table)
            print()
    else:
        for cat, cat_keywords in sorted(by_category.items()):
            print(f"\n{cat.upper()} ({len(cat_keywords)})")
            print("-" * 40)
            for kw in cat_keywords:
                if verbose:
                    print(f"  {kw.id} ({kw.occurrences}x)")
                    print(f"    Variations: {', '.join(kw.match[:5])}")
                else:
                    print(f"  {kw.id} ({kw.occurrences}x)")


@app.command("add")
def add_keyword(
    keyword_id: str = typer.Argument(..., help="Keyword identifier (e.g., 'TC-1001')"),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses default if not specified.",
    ),
    category: str = typer.Option(
        "custom",
        "--category",
        help="Category for the keyword.",
    ),
    variations: Optional[str] = typer.Option(
        None,
        "--variations",
        "-v",
        help="Comma-separated list of variations.",
    ),
) -> None:
    """Add a custom keyword to the vocabulary."""
    from fitz_ai.retrieval.vocabulary import Keyword, VocabularyStore, generate_variations

    collection = _get_collection(collection)
    store = VocabularyStore(collection=collection)

    # Generate default variations
    default_variations = generate_variations(keyword_id, category)

    # Add custom variations if provided
    if variations:
        custom = [v.strip() for v in variations.split(",") if v.strip()]
        default_variations = sorted(set(default_variations + custom), key=str.lower)

    keyword = Keyword(
        id=keyword_id,
        category=category,
        match=default_variations,
        occurrences=0,
        user_defined=True,
        auto_generated=default_variations.copy(),
    )

    store.add_keyword(keyword)

    ui.success(
        f"Added keyword '{keyword_id}' to collection '{collection}' with {len(default_variations)} variations."
    )
    if RICH:
        console.print(f"[dim]Variations: {', '.join(default_variations[:5])}[/dim]")


@app.command("edit")
def edit_keyword(
    keyword_id: str = typer.Argument(..., help="Keyword identifier to edit"),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses default if not specified.",
    ),
    add: Optional[str] = typer.Option(
        None,
        "--add",
        "-a",
        help="Add a variation to the keyword.",
    ),
    remove: Optional[str] = typer.Option(
        None,
        "--remove",
        "-r",
        help="Remove a variation from the keyword.",
    ),
) -> None:
    """Edit variations for an existing keyword."""
    from fitz_ai.retrieval.vocabulary import VocabularyStore

    if not add and not remove:
        ui.error("Specify --add or --remove to edit variations.")
        raise typer.Exit(1)

    collection = _get_collection(collection)
    store = VocabularyStore(collection=collection)
    keywords = store.load()

    # Find keyword
    target = None
    for kw in keywords:
        if kw.id.lower() == keyword_id.lower():
            target = kw
            break

    if not target:
        ui.error(f"Keyword '{keyword_id}' not found in collection '{collection}'.")
        raise typer.Exit(1)

    if add:
        if add not in target.match:
            target.match.append(add)
            ui.success(f"Added variation '{add}' to {keyword_id}.")
        else:
            ui.warning(f"Variation '{add}' already exists.")

    if remove:
        if remove in target.match:
            target.match.remove(remove)
            ui.success(f"Removed variation '{remove}' from {keyword_id}.")
        else:
            ui.warning(f"Variation '{remove}' not found.")

    store.save(keywords)


@app.command("remove")
def remove_keyword(
    keyword_id: str = typer.Argument(..., help="Keyword identifier to remove"),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses default if not specified.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Remove a keyword from the vocabulary."""
    from fitz_ai.retrieval.vocabulary import VocabularyStore

    collection = _get_collection(collection)
    store = VocabularyStore(collection=collection)

    if not store.exists():
        ui.error(f"No vocabulary file found for collection '{collection}'.")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Remove keyword '{keyword_id}'?")
        if not confirm:
            ui.info("Cancelled.")
            return

    if store.remove_keyword(keyword_id):
        ui.success(f"Removed keyword '{keyword_id}' from collection '{collection}'.")
    else:
        ui.error(f"Keyword '{keyword_id}' not found.")
        raise typer.Exit(1)


@app.command("suggest")
def suggest_keywords(
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection to scan. Uses default if not specified.",
    ),
    min_occurrences: int = typer.Option(
        2,
        "--min",
        "-m",
        help="Minimum occurrences to suggest.",
    ),
) -> None:
    """Suggest keywords from the corpus."""
    from fitz_ai.cli.context import CLIContext
    from fitz_ai.retrieval.vocabulary import KeywordDetector, VocabularyStore
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    ctx = CLIContext.load()

    if collection is None:
        collection = ctx.retrieval_collection

    ui.info(f"Scanning collection '{collection}' for keywords...")

    # Load chunks from vector DB
    try:
        vector_client = get_vector_db_plugin(ctx.vector_db_plugin)
        chunks = vector_client.get_all_chunks(collection)
    except Exception as e:
        ui.error(f"Failed to load chunks: {e}")
        raise typer.Exit(1)

    if not chunks:
        ui.warning("No chunks found in collection.")
        return

    ui.info(f"Scanning {len(chunks)} chunks...")

    # Detect keywords
    detector = KeywordDetector(min_occurrences=min_occurrences)
    keywords = detector.detect_from_chunks(chunks)

    if not keywords:
        ui.info("No keywords detected with current settings.")
        ui.info(f"Try lowering --min (currently {min_occurrences}).")
        return

    # Show suggestions
    print()
    ui.success(f"Found {len(keywords)} potential keywords:")
    print()

    # Group by category
    by_category: dict[str, list] = {}
    for kw in keywords:
        if kw.category not in by_category:
            by_category[kw.category] = []
        by_category[kw.category].append(kw)

    if RICH:
        from rich.table import Table

        for cat, cat_keywords in sorted(by_category.items()):
            table = Table(title=f"{cat.upper()}", show_header=True, header_style="bold")
            table.add_column("ID", style="cyan")
            table.add_column("Occurrences", justify="right")
            table.add_column("First Seen", style="dim")

            for kw in cat_keywords[:10]:  # Limit to 10 per category
                table.add_row(kw.id, str(kw.occurrences), kw.first_seen or "-")

            if len(cat_keywords) > 10:
                table.add_row(f"... +{len(cat_keywords) - 10} more", "", "")

            console.print(table)
            print()
    else:
        for cat, cat_keywords in sorted(by_category.items()):
            print(f"\n{cat.upper()}:")
            for kw in cat_keywords[:10]:
                print(f"  {kw.id} ({kw.occurrences}x)")

    # Prompt to save
    print()
    save = typer.confirm("Save these keywords to vocabulary?")
    if save:
        store = VocabularyStore(collection=collection)
        store.merge_and_save(keywords, source_docs=len(chunks))
        ui.success(f"Saved {len(keywords)} keywords to {store.path}")
    else:
        ui.info("Keywords not saved.")


@app.command("detect")
def detect_keywords(
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection to scan. Uses default if not specified.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing vocabulary (preserves user modifications).",
    ),
) -> None:
    """Re-detect keywords from collection and update vocabulary."""
    from fitz_ai.cli.context import CLIContext
    from fitz_ai.retrieval.vocabulary import KeywordDetector, VocabularyStore
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    ctx = CLIContext.load()

    if collection is None:
        collection = ctx.retrieval_collection

    store = VocabularyStore(collection=collection)

    if store.exists() and not force:
        ui.warning(
            "Vocabulary file exists. Use --force to re-detect (user modifications preserved)."
        )
        return

    ui.info(f"Detecting keywords from collection '{collection}'...")

    # Load chunks from vector DB
    try:
        vector_client = get_vector_db_plugin(ctx.vector_db_plugin)
        chunks = vector_client.get_all_chunks(collection)
    except Exception as e:
        ui.error(f"Failed to load chunks: {e}")
        raise typer.Exit(1)

    if not chunks:
        ui.warning("No chunks found in collection.")
        return

    ui.info(f"Scanning {len(chunks)} chunks...")

    # Detect keywords
    detector = KeywordDetector()
    keywords = detector.detect_from_chunks(chunks)

    if not keywords:
        ui.info("No keywords detected.")
        return

    # Save (merge with existing to preserve user modifications)
    merged = store.merge_and_save(keywords, source_docs=len(chunks))

    ui.success(f"Detected {len(keywords)} keywords, saved {len(merged)} total to {store.path}")

    # Show summary by category
    by_category: dict[str, int] = {}
    for kw in merged:
        by_category[kw.category] = by_category.get(kw.category, 0) + 1

    print()
    for cat, count in sorted(by_category.items()):
        ui.info(f"  {cat}: {count}")


# Export command function for main CLI
def command() -> None:
    """Run the keywords command group."""
    app()
