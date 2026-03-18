# tests/fixtures/__init__.py
"""
Test fixture builders for reducing mock usage.

Import commonly used builders for easy access:
    from tests.fixtures import (
        SourceFileBuilder,
        DocumentElementBuilder,
        make_parsed_doc,
        make_test_stores
    )
"""

from .builders import (
    AddressBuilder,
    CollectionInfoBuilder,
    DocumentElementBuilder,
    InMemoryImportStore,
    InMemoryRawStore,
    InMemorySymbolStore,
    ParsedDocumentBuilder,
    ReadResultBuilder,
    SourceFileBuilder,
    make_parsed_doc,
    make_symbol_result,
    make_test_stores,
    make_text_file,
)

__all__ = [
    # Builders
    "SourceFileBuilder",
    "DocumentElementBuilder",
    "ParsedDocumentBuilder",
    "AddressBuilder",
    "ReadResultBuilder",
    "CollectionInfoBuilder",
    # Stores
    "InMemoryRawStore",
    "InMemorySymbolStore",
    "InMemoryImportStore",
    # Factory functions
    "make_text_file",
    "make_parsed_doc",
    "make_symbol_result",
    "make_test_stores",
]
