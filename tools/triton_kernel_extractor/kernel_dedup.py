"""Deduplicate Triton kernels by source content across extracted samples.

Reads the output of the ``extract`` pipeline step (``triton_kernel/*.py``
files under each sample directory) and identifies duplicate kernels — kernels
with identical source code that appear in multiple samples.

This is distinct from ``graph_hash.txt``-based dedup which identifies
identical *subgraphs*.  Kernel-level dedup identifies identical *compiled
kernels* produced by TorchInductor, which may come from different subgraphs
that share the same operator patterns (e.g., two different backward graphs
both containing ``layer_norm_bwd``).

Usage (as subcommand)::

    python3 -m tools.triton_kernel_extractor dedup \\
        --input-dir /data/output/extracted \\
        --output /tmp/dedup_report.json

Usage (standalone)::

    python3 tools/triton_kernel_extractor/kernel_dedup.py \\
        --input-dir /data/output/extracted \\
        --output /tmp/dedup_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_kernel_source(source: str) -> str:
    """Strip comments and blank lines to produce a normalized form for hashing."""
    lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _hash_source(normalized: str) -> str:
    """Compute a stable MD5 hash of the normalized kernel source."""
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def dedup_kernels(input_dir: Path) -> dict:
    """Walk *input_dir* for ``triton_kernel/*.py`` files and compute dedup stats.

    Parameters
    ----------
    input_dir:
        Root directory containing per-sample subdirectories.  Each sample
        directory is expected to have a ``triton_kernel/`` subdirectory
        with ``.py`` files produced by the ``extract`` pipeline step.

    Returns
    -------
    dict with keys:
        - ``total_samples``: number of sample directories scanned
        - ``total_kernel_instances``: total ``.py`` files found
        - ``unique_kernel_hashes``: number of distinct (normalized) source hashes
        - ``dedup_rate_percent``: ``(1 - unique / total) * 100``
        - ``avg_kernels_per_sample``: mean kernel count per sample
        - ``kernel_name_freq``: ``{kernel_name: occurrence_count}`` across all samples
        - ``per_sample``: list of ``{path, kernel_count, hashes}`` per sample
    """
    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        return {}

    # Enumerate sample directories (any dir containing triton_kernel/).
    samples: list[Path] = []
    for dirpath, dirnames, _ in os.walk(str(input_dir)):
        if "triton_kernel" in dirnames:
            samples.append(Path(dirpath))

    if not samples:
        logger.warning("No samples with triton_kernel/ found under %s", input_dir)
        return {}

    all_hashes: list[str] = []
    kernel_name_counter: Counter[str] = Counter()
    per_sample: list[dict] = []

    for sample_dir in sorted(samples):
        kernel_dir = sample_dir / "triton_kernel"
        kernel_files = sorted(kernel_dir.glob("*.py"))
        hashes: list[str] = []
        for kf in kernel_files:
            try:
                source = kf.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.warning("Cannot read kernel file: %s", kf)
                continue
            normalized = _normalize_kernel_source(source)
            h = _hash_source(normalized)
            hashes.append(h)
            all_hashes.append(h)
            kernel_name_counter[kf.stem] += 1

        per_sample.append(
            {
                "sample": str(sample_dir),
                "kernel_count": len(hashes),
                "hashes": hashes,
            }
        )

    total = len(all_hashes)
    unique = len(set(all_hashes))
    dedup_rate = round((1 - unique / total) * 100, 2) if total > 0 else 0.0
    avg = round(total / len(samples), 2) if samples else 0.0

    return {
        "total_samples": len(samples),
        "total_kernel_instances": total,
        "unique_kernel_hashes": unique,
        "dedup_rate_percent": dedup_rate,
        "avg_kernels_per_sample": avg,
        "kernel_name_freq": dict(
            kernel_name_counter.most_common()
        ),
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate Triton kernels by source content.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Root directory containing per-sample subdirectories with "
            "triton_kernel/*.py files (output of the extract pipeline step)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for the dedup report.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )

    logger.info("Scanning: %s", args.input_dir)
    report = dedup_kernels(args.input_dir)
    if not report:
        logger.warning("No data to report.")
        return

    # Print summary.
    print("\n=== Kernel Dedup Report ===")
    print(f"Total samples scanned:       {report['total_samples']}")
    print(f"Total kernel instances:       {report['total_kernel_instances']}")
    print(f"Unique kernel hashes:         {report['unique_kernel_hashes']}")
    print(f"Dedup rate:                   {report['dedup_rate_percent']}%")
    print(f"Avg kernels per sample:       {report['avg_kernels_per_sample']}")
    print(f"\nTop kernel types by frequency:")
    for name, count in list(report["kernel_name_freq"].items())[:20]:
        bar = "#" * min(count, 40)
        print(f"  {name:50s} {count:4d}  {bar}")

    # Write JSON.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()