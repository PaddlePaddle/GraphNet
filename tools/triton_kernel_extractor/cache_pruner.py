"""Prune large TorchInductor cache artifacts while preserving later workflows."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .config import is_sample_dir

logger = logging.getLogger(__name__)
_MAKE_RANGE_END_PATTERN = re.compile(r"tt\.make_range\s*\{end = (\d+)")

# Files needed to keep later extraction and analysis working.
_KEEP_FILENAMES = {
    "test_compiler_log.log",
    "graph_hash.txt",
    "model.py",
    "output_code.py",
}


def _should_keep_file(path: Path) -> bool:
    """Return True if *path* is required by later extract/analyze steps."""
    name = path.name
    if name in _KEEP_FILENAMES:
        return True
    if name.endswith(".best_config"):
        return True
    if name.endswith(".ptx"):
        return True
    if name.endswith(".json"):
        return True
    return False


def _write_pruned_triton_metadata(sample_dir: Path) -> None:
    """Persist compact metadata needed after deleting bulky Triton sources."""
    for source_path in sample_dir.rglob("*.source"):
        ptx_path = source_path.with_suffix(".ptx")
        if not ptx_path.is_file():
            continue
        try:
            content = source_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        block_values = [int(m) for m in _MAKE_RANGE_END_PATTERN.findall(content)]
        if not block_values:
            continue
        meta_path = source_path.with_suffix(".pruned_meta.json")
        try:
            meta_path.write_text(
                json.dumps({"block_values": block_values}, separators=(",", ":")),
                encoding="utf-8",
            )
        except OSError:
            logger.debug("Cannot write pruned Triton metadata: %s", meta_path)


def _looks_like_sample_cache_dir(directory: Path) -> bool:
    """Heuristic for directories that contain a compiled sample cache."""
    if not directory.is_dir():
        return False
    if (directory / "test_compiler_log.log").is_file():
        return True
    if (directory / "original_graph").is_dir():
        return True
    if (directory / "triton").is_dir():
        return True
    return False


def _prune_one_sample(sample_dir: Path) -> tuple[int, int]:
    """Delete unneeded files from a single sample cache directory."""
    removed_files = 0
    removed_dirs = 0

    _write_pruned_triton_metadata(sample_dir)

    for path in sorted(sample_dir.rglob("*"), reverse=True):
        if path.is_file():
            if not _should_keep_file(path):
                try:
                    path.unlink()
                    removed_files += 1
                except FileNotFoundError:
                    pass

    # Remove now-empty directories bottom-up, but keep the sample root itself.
    for path in sorted(sample_dir.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
                removed_dirs += 1
            except OSError:
                pass

    return removed_files, removed_dirs


def prune_compilation_cache(cache_dir: Path) -> tuple[int, int]:
    """Prune large intermediate artifacts from compiled sample caches.

    This keeps only the files required for later extraction and analysis:
    ``test_compiler_log.log``, ``original_graph/{model.py,graph_hash.txt}``,
    ``output_code.py``, ``*.best_config``, ``*.ptx``, and lightweight Triton
    metadata files (``*.json`` / ``*.pruned_meta.json``) used to disambiguate
    autotuning candidates when ``triton_cache_hash`` is unavailable.
    """
    if not cache_dir.is_dir():
        logger.warning("Cache directory does not exist, skipping: %s", cache_dir)
        return 0, 0

    total_files = 0
    total_dirs = 0

    # Support pruning a single sample cache directory directly.
    if _looks_like_sample_cache_dir(cache_dir):
        removed_files, removed_dirs = _prune_one_sample(cache_dir)
        logger.info(
            "Pruned %s: -%d files, -%d dirs",
            cache_dir,
            removed_files,
            removed_dirs,
        )
        logger.info(
            "Prune complete: removed %d files and %d directories under %s",
            removed_files,
            removed_dirs,
            cache_dir,
        )
        return removed_files, removed_dirs

    # Prune sample caches in the root, kept/, and discarded/ trees.
    roots = [cache_dir, cache_dir / "kept", cache_dir / "discarded"]
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if root == cache_dir and not _looks_like_sample_cache_dir(child):
                continue
            if root != cache_dir and not is_sample_dir(child.name):
                continue

            removed_files, removed_dirs = _prune_one_sample(child)
            if removed_files or removed_dirs:
                logger.info(
                    "Pruned %s: -%d files, -%d dirs",
                    child.relative_to(cache_dir),
                    removed_files,
                    removed_dirs,
                )
            total_files += removed_files
            total_dirs += removed_dirs

    logger.info(
        "Prune complete: removed %d files and %d directories under %s",
        total_files,
        total_dirs,
        cache_dir,
    )
    return total_files, total_dirs
