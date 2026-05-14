"""Pipeline configuration types and constants."""

from __future__ import annotations

import dataclasses
from pathlib import Path

# Log patterns emitted by graph_net_bench.torch.test_compiler with --kernel-time.
SPEEDUP_KERNEL_PATTERN = r"\[Speedup\]\[kernel\]:\s*([\d.]+)"
SPEEDUP_E2E_PATTERN = r"\[Speedup\]\[e2e\]:\s*([\d.]+)"

# Subdirectory names reserved for internal bookkeeping inside the cache directory.
# These are skipped when iterating over sample directories.
RESERVED_DIR_NAMES = frozenset({"kept", "discarded"})

# Prefix used by temporary pipeline artifacts (chunk files, worker logs, sample
# lists).  Directories whose name starts with this prefix are skipped during
# sample iteration.
RESERVED_DIR_PREFIX = "_"

# Minimum kernel speedup required to keep a compiled sample.
SPEEDUP_THRESHOLD = 1.0


def is_sample_dir(name: str) -> bool:
    """Return True if *name* is a real sample directory, not a reserved one.

    Filters out ``kept``, ``discarded``, and directories starting with ``_``
    (temporary pipeline artifacts such as chunk files and worker logs).
    """
    if name in RESERVED_DIR_NAMES:
        return False
    if name.startswith(RESERVED_DIR_PREFIX):
        return False
    return True


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    """Immutable top-level configuration for a single pipeline run."""

    gpu_ids: list[int]
    graph_dir: Path
    output_dir: Path
    allow_list: Path | None = None
    max_autotune_no_cudagraphs: bool = False
