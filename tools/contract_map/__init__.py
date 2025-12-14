# tools/contract_map/__init__.py
"""
Contract Map Generator - A modular tool for analyzing Python codebases.

This package provides utilities to extract and visualize:
- Models and protocols
- Plugin registries
- Import graphs and layering violations
- Configuration surfaces
- Code statistics and hotspots
"""

from .analysis import (
    compute_config_surface,
    compute_hotspots,
    compute_invariants,
    compute_stats,
    discover_entrypoints,
)
from .common import (
    REPO_ROOT,
    ContractMap,
    HealthIssue,
    ImportGraph,
    ModelContract,
    ProtocolContract,
    RegistryContract,
)
from .discovery import scan_all_discoveries
from .imports import build_import_graph
from .models import extract_models, extract_protocols
from .registries import extract_registries

__all__ = [
    "ContractMap",
    "ModelContract",
    "ProtocolContract",
    "RegistryContract",
    "ImportGraph",
    "HealthIssue",
    "REPO_ROOT",
    "extract_models",
    "extract_protocols",
    "extract_registries",
    "build_import_graph",
    "scan_all_discoveries",
    "discover_entrypoints",
    "compute_hotspots",
    "compute_stats",
    "compute_config_surface",
    "compute_invariants",
]

__version__ = "1.0.0"
