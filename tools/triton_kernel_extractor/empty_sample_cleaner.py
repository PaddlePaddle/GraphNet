"""Step 5: Remove output samples that have no extracted triton kernels."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def clean_empty_kernel_samples(output_dir: Path) -> tuple[int, int]:
    """Delete samples that contain ``original_graph/`` but no ``triton_kernel/``.

    Returns:
        A tuple of ``(removed_count, kept_count)``.
    """
    if not output_dir.is_dir():
        logger.warning("Output directory does not exist: %s", output_dir)
        return 0, 0

    total = 0
    removed = 0

    for sample_dir in sorted(output_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        total += 1

        has_graph = (sample_dir / "original_graph").is_dir()
        has_kernel = (sample_dir / "triton_kernel").is_dir()

        if has_graph and not has_kernel:
            logger.info("  Removing (no triton_kernel): %s", sample_dir.name)
            shutil.rmtree(sample_dir)
            removed += 1

    kept = total - removed
    logger.info(
        "Cleanup: %d removed (no triton_kernel), %d kept (total: %d)",
        removed,
        kept,
        total,
    )
    return removed, kept
