# fitz/core/_contracts/__init__.py
"""
Internal architectural contracts for Fitz.

This package defines *abstract* invariants that describe how the system
is allowed to be structured. It is intentionally internal, non-runtime,
and non-user-facing.

Runtime code MUST NOT import from this package.
"""
