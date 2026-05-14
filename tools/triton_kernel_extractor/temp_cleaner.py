"""Step 3: Remove Python bytecode caches from a directory tree."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def clean_temp_files(directory: Path) -> None:
    """Recursively delete ``__pycache__`` dirs, ``*.pyc``, and ``*.pyo`` files.

    Mirrors the original bash implementation::

        find "$dir" -type d -name "__pycache__" -exec rm -rf {} +
        find "$dir" -type f -name "*.pyc" -delete
        find "$dir" -type f -name "*.pyo" -delete

    Silently skips entries that vanish between discovery and deletion (e.g.
    a ``.pyc`` inside a ``__pycache__`` that was already removed).
    """
    if not directory.is_dir():
        logger.warning("Directory does not exist, skipping: %s", directory)
        return

    logger.info("Cleaning temp files from %s ...", directory)

    # Collect first, then delete — avoids mutating the tree during iteration.
    pycache_dirs = sorted(directory.rglob("__pycache__"), reverse=True)
    for d in pycache_dirs:
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)

    for pattern in ("*.pyc", "*.pyo"):
        for f in directory.rglob(pattern):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
