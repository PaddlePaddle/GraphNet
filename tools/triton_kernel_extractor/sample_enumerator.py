"""Enumerate graph samples for the extraction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def enumerate_list_samples(allow_list: Path, graph_dir: Path) -> list[str]:
    """Read relative sample paths from *allow_list* and resolve against *graph_dir*.

    Returns absolute paths to sample directories.  Blank lines are skipped.

    Raises:
        FileNotFoundError: If *allow_list* does not exist.
    """
    paths: list[str] = []
    with open(allow_list, encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if stripped:
                paths.append(str(graph_dir / stripped))
    return paths


def enumerate_graph_dir(graph_dir: Path) -> list[str]:
    """Discover samples by recursively scanning *graph_dir* for ``model.py``.

    Returns the sorted list of parent directories that contain a ``model.py``.

    Raises:
        FileNotFoundError: If *graph_dir* does not exist.
    """
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
    parents = sorted(
        {str(p.parent) for p in graph_dir.rglob("model.py") if p.is_file()}
    )
    return parents


def compute_unique_dir(sample_path: str, graph_dir: str) -> str:
    """Derive a flat directory name from a sample path.

    Uses the relative portion below *graph_dir*, with ``/`` replaced by ``_``.
    If *sample_path* is not under *graph_dir*, the full path is flattened
    with leading slashes stripped to avoid reserved-prefix collisions.
    """
    prefix = graph_dir.rstrip("/") + "/"
    if sample_path.startswith(prefix):
        rel = sample_path[len(prefix) :]
    else:
        rel = sample_path.lstrip("/")
    return rel.replace("/", "_")
