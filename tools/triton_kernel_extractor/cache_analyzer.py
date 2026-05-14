"""Analyze an inductor cache directory and produce summary statistics and plots.

This module replaces the standalone ``analyze_inductor_cache.sh`` script.  It
concatenates compiler logs from all sample states (root, kept, discarded),
computes speedup statistics, and generates distribution plots.

The analysis can be invoked via the CLI::

    python3 -m tools.triton_kernel_extractor analyze <cache_dir> [--output-dir DIR]
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import SPEEDUP_E2E_PATTERN, SPEEDUP_KERNEL_PATTERN, is_sample_dir

logger = logging.getLogger(__name__)

_SPEEDUP_E2E_RE = re.compile(SPEEDUP_E2E_PATTERN)
_SPEEDUP_KERNEL_RE = re.compile(SPEEDUP_KERNEL_PATTERN)


# ---------------------------------------------------------------------------
# Step 1: Log concatenation
# ---------------------------------------------------------------------------


def _concat_logs(search_dir: Path) -> tuple[str, int]:
    """Concatenate ``test_compiler_log.log`` from all samples under *search_dir*.

    Returns the combined text and the number of log files found.
    """
    if not search_dir.is_dir():
        return "", 0

    parts: list[str] = []
    count = 0
    for sample_dir in sorted(search_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        if not is_sample_dir(sample_dir.name):
            continue
        log_file = sample_dir / "test_compiler_log.log"
        if log_file.is_file():
            parts.append(log_file.read_text(encoding="utf-8", errors="replace"))
            count += 1
    return "\n".join(parts), count


def concatenate_logs(cache_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Concatenate logs from root, kept, and discarded sample directories.

    Writes three files to *output_dir* and returns their paths:
    ``(all_log, kept_log, discarded_log)``.
    """
    root_text, root_count = _concat_logs(cache_dir)
    kept_text, kept_count = _concat_logs(cache_dir / "kept")
    discarded_text, discarded_count = _concat_logs(cache_dir / "discarded")

    all_text = "\n".join(filter(None, [root_text, kept_text, discarded_text]))

    all_log = output_dir / "all_samples.log"
    kept_log = output_dir / "kept_samples.log"
    discarded_log = output_dir / "discarded_samples.log"

    all_log.write_text(all_text, encoding="utf-8")
    kept_log.write_text(kept_text, encoding="utf-8")
    discarded_log.write_text(discarded_text, encoding="utf-8")

    total = root_count + kept_count + discarded_count
    logger.info(
        "  Logs concatenated: %d total (%d root, %d kept, %d discarded)",
        total,
        root_count,
        kept_count,
        discarded_count,
    )
    logger.info("  All:       %s", all_log)
    logger.info("  Kept:      %s", kept_log)
    logger.info("  Discarded: %s", discarded_log)

    return all_log, kept_log, discarded_log


# ---------------------------------------------------------------------------
# Step 2: Summary statistics
# ---------------------------------------------------------------------------


def _parse_speedups(text: str, pattern: re.Pattern[str]) -> list[float]:
    """Extract all speedup values matching *pattern* from log text."""
    return [float(m) for m in pattern.findall(text)]


def _percentile(values: list[float], p: float) -> float:
    """Compute the *p*-th percentile (0–100) of a sorted list."""
    if not values:
        return 0.0
    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(values) else f
    return values[f] + (k - f) * (values[c] - values[f])


def _format_speedup_stats(values: list[float], label: str) -> str:
    """Format a block of descriptive statistics for a speedup distribution."""
    lines: list[str] = []
    n = len(values)
    lines.append(f"  Samples with {label} speedup: {n}")

    if n == 0:
        return "\n".join(lines)

    values_sorted = sorted(values)
    mean = sum(values_sorted) / n
    median = _percentile(values_sorted, 50)

    lines.append("")
    lines.append(f"  Mean:       {mean:.4f}")
    lines.append(f"  Median:     {median:.4f}")
    lines.append(f"  Min:        {values_sorted[0]:.4f}")
    lines.append(f"  Max:        {values_sorted[-1]:.4f}")
    lines.append(f"  P5:         {_percentile(values_sorted, 5):.4f}")
    lines.append(f"  P25:        {_percentile(values_sorted, 25):.4f}")
    lines.append(f"  P75:        {_percentile(values_sorted, 75):.4f}")
    lines.append(f"  P95:        {_percentile(values_sorted, 95):.4f}")

    ge2 = sum(1 for v in values_sorted if v >= 2.0)
    ge1_5 = sum(1 for v in values_sorted if v >= 1.5)
    ge1 = sum(1 for v in values_sorted if v >= 1.0)
    lt1 = sum(1 for v in values_sorted if v < 1.0)
    lt0_5 = sum(1 for v in values_sorted if v < 0.5)

    lines.append("")
    lines.append(f"  Speedup >= 2.0:  {ge2} ({ge2/n*100:.1f}%)")
    lines.append(f"  Speedup >= 1.5:  {ge1_5} ({ge1_5/n*100:.1f}%)")
    lines.append(f"  Speedup >= 1.0:  {ge1} ({ge1/n*100:.1f}%)")
    lines.append(f"  Speedup <  1.0:  {lt1} ({lt1/n*100:.1f}%) [negative optimization]")
    lines.append(f"  Speedup <  0.5:  {lt0_5} ({lt0_5/n*100:.1f}%) [severe regression]")

    return "\n".join(lines)


def _count_subdirs(directory: Path) -> int:
    """Count immediate subdirectories of *directory*."""
    if not directory.is_dir():
        return 0
    return sum(1 for d in directory.iterdir() if d.is_dir())


def generate_summary(
    cache_dir: Path,
    all_log_text: str,
    discarded_log_text: str,
    output_dir: Path,
) -> Path:
    """Generate a text summary report and return its path."""
    kernel_speedups = _parse_speedups(all_log_text, _SPEEDUP_KERNEL_RE)
    e2e_speedups = _parse_speedups(all_log_text, _SPEEDUP_E2E_RE)

    # Count samples in each state.
    root_samples = (
        sum(1 for d in cache_dir.iterdir() if d.is_dir() and is_sample_dir(d.name))
        if cache_dir.is_dir()
        else 0
    )
    kept_samples = _count_subdirs(cache_dir / "kept")
    discarded_samples = _count_subdirs(cache_dir / "discarded")
    total_samples = root_samples + kept_samples + discarded_samples

    def pct(n: int) -> str:
        return f"{n/total_samples*100:.1f}" if total_samples > 0 else "0.0"

    lines: list[str] = [
        "Inductor Cache Analysis Report",
        "==============================",
        f"Cache dir: {cache_dir}",
        f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Sample Counts",
        "-------------",
        f"  Total:     {total_samples}",
        f"  Kept:      {kept_samples} ({pct(kept_samples)}%)",
        f"  Discarded: {discarded_samples} ({pct(discarded_samples)}%)",
        f"  Unclassified (root): {root_samples}",
        "",
        "Kernel Speedup Distribution (primary — used for kept/discarded filtering)",
        "-------------------------------------------------------------------------",
        _format_speedup_stats(kernel_speedups, "Kernel"),
        "",
        "E2E Speedup Distribution (secondary — includes framework overhead)",
        "-------------------------------------------------------------------",
        _format_speedup_stats(e2e_speedups, "E2E"),
    ]

    # Failure analysis from discarded logs.
    if discarded_log_text:
        neg_opt = sum(1 for v in kernel_speedups if 0 < v < 1.0)
        error_lines = sum(
            1
            for line in discarded_log_text.splitlines()
            if re.search(r"ERROR|Exception|Traceback", line)
        )
        lines.extend(
            [
                "",
                "Failure/Discard Analysis",
                "------------------------",
                f"  Negative optimization (0 < kernel speedup < 1): {neg_opt}",
                f"  Logs with errors/exceptions: {error_lines} lines",
            ]
        )

    report = "\n".join(lines) + "\n"
    summary_file = output_dir / "summary.txt"
    summary_file.write_text(report, encoding="utf-8")

    # Also print to console.
    logger.info("%s", report)
    logger.info("Summary saved to: %s", summary_file)

    return summary_file


# ---------------------------------------------------------------------------
# Step 3: Plots
# ---------------------------------------------------------------------------


def _check_plotting_deps() -> bool:
    """Return True if matplotlib and numpy are importable."""
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401

        return True
    except ImportError:
        return False


def generate_plots(
    all_log_text: str,
    all_log_path: Path,
    output_dir: Path,
) -> None:
    """Generate speedup distribution plots.

    Produces:
      - ``speedup_histogram.png`` — raw and log2 histograms of kernel speedup.
      - ``speedup_cdf.png`` — cumulative distribution function.
      - ``violin.png`` — via ``graph_net_visual.plot_violin`` (best effort).
      - ``ESt.png`` — via ``graph_net_visual.plot_ESt`` (best effort).
    """
    if not _check_plotting_deps():
        logger.warning(
            "  Skipping plots: matplotlib or numpy not installed. "
            "Run: pip install matplotlib numpy"
        )
        return

    kernel_speedups = _parse_speedups(all_log_text, _SPEEDUP_KERNEL_RE)
    if not kernel_speedups:
        logger.info("  No kernel speedup data for plotting.")
        return

    _generate_builtin_plots(kernel_speedups, output_dir)
    _run_visual_plot("graph_net_visual.plot_violin", all_log_path, output_dir)
    _run_visual_plot(
        "graph_net_visual.plot_ESt",
        all_log_path,
        output_dir,
        extra_args=["--disable-aggregation-mode"],
    )


def _generate_builtin_plots(
    kernel_speedups: list[float],
    output_dir: Path,
) -> None:
    """Generate histogram and CDF plots using matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    speedups = np.array(kernel_speedups)

    # --- Histogram: raw + log2 side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax1 = axes[0]
    bins = np.concatenate(
        [
            np.arange(0, 1.0, 0.1),
            np.arange(1.0, 2.0, 0.1),
            np.arange(2.0, max(5.0, float(np.percentile(speedups, 99))) + 0.5, 0.5),
        ]
    )
    ax1.hist(speedups, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax1.axvline(
        x=1.0, color="red", linestyle="--", linewidth=1.5, label="speedup = 1.0"
    )
    ax1.axvline(
        x=float(np.median(speedups)),
        color="orange",
        linestyle="-",
        linewidth=1.5,
        label=f"median = {np.median(speedups):.3f}",
    )
    ax1.set_xlabel("Kernel Speedup", fontsize=14)
    ax1.set_ylabel("Count", fontsize=14)
    ax1.set_title("Kernel Speedup Distribution", fontsize=16)
    ax1.legend(fontsize=12)

    n_total = len(speedups)
    n_pos = int(np.sum(speedups >= 1.0))
    n_neg = int(np.sum(speedups < 1.0))
    stats_text = (
        f"Total: {n_total}\n"
        f"Speedup >= 1: {n_pos} ({n_pos/n_total*100:.1f}%)\n"
        f"Speedup < 1: {n_neg} ({n_neg/n_total*100:.1f}%)\n"
        f"Mean: {np.mean(speedups):.3f}\n"
        f"Median: {np.median(speedups):.3f}"
    )
    ax1.text(
        0.97,
        0.97,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Log2 histogram.
    positive = speedups[speedups > 0]
    log2_sp = np.log2(positive)
    ax2 = axes[1]
    ax2.hist(log2_sp, bins=60, color="darkorange", edgecolor="white", alpha=0.85)
    ax2.axvline(
        x=0, color="red", linestyle="--", linewidth=1.5, label="log2(speedup) = 0"
    )
    ax2.axvline(
        x=float(np.median(log2_sp)),
        color="blue",
        linestyle="-",
        linewidth=1.5,
        label=f"median = {np.median(log2_sp):.3f}",
    )
    ax2.set_xlabel("log2(Kernel Speedup)", fontsize=14)
    ax2.set_ylabel("Count", fontsize=14)
    ax2.set_title("log2(Kernel Speedup) Distribution", fontsize=16)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    hist_path = output_dir / "speedup_histogram.png"
    plt.savefig(str(hist_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", hist_path)

    # --- CDF ---
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    sorted_sp = np.sort(speedups)
    cdf = np.arange(1, len(sorted_sp) + 1) / len(sorted_sp)
    ax3.plot(sorted_sp, cdf, color="steelblue", linewidth=2)
    ax3.axvline(
        x=1.0, color="red", linestyle="--", linewidth=1.2, label="speedup = 1.0"
    )

    cdf_at_1 = float(np.searchsorted(sorted_sp, 1.0) / len(sorted_sp))
    ax3.axhline(y=cdf_at_1, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax3.plot(1.0, cdf_at_1, "ro", markersize=8)
    ax3.text(1.05, cdf_at_1, f"{cdf_at_1*100:.1f}% below 1.0", fontsize=12, color="red")
    ax3.set_xlabel("Kernel Speedup", fontsize=14)
    ax3.set_ylabel("Cumulative Fraction", fontsize=14)
    ax3.set_title("Kernel Speedup CDF", fontsize=16)
    ax3.set_xlim(0, min(5.0, float(np.percentile(speedups, 99.5))))
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)

    cdf_path = output_dir / "speedup_cdf.png"
    plt.savefig(str(cdf_path), dpi=200, bbox_inches="tight")
    plt.close(fig2)
    logger.info("  Saved: %s", cdf_path)


def _run_visual_plot(
    module: str,
    log_path: Path,
    output_dir: Path,
    *,
    extra_args: list[str] | None = None,
) -> None:
    """Run a graph_net_visual plotting module (best effort, never fatal)."""
    cmd = [
        sys.executable,
        "-m",
        module,
        "--benchmark-path",
        str(log_path),
        "--output-dir",
        str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        # Log any saved/error lines from stdout+stderr.
        for line in (result.stdout + result.stderr).splitlines():
            if re.search(r"saved|Saved|Error|Warning", line, re.IGNORECASE):
                logger.info("  %s: %s", module.rsplit(".", 1)[-1], line.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("  %s unavailable: %s", module, exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_cache(cache_dir: Path, output_dir: Path) -> None:
    """Run the full analysis pipeline on an inductor cache directory."""
    if not cache_dir.is_dir():
        logger.error("Cache directory does not exist: %s", cache_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("======================================================")
    logger.info(" Inductor Cache Analysis")
    logger.info(" Input:  %s", cache_dir)
    logger.info(" Output: %s", output_dir)
    logger.info("======================================================")

    # Step 1: Concatenate logs.
    logger.info("")
    logger.info("=== Step 1: Concatenating log files ===")
    all_log, _kept_log, discarded_log = concatenate_logs(cache_dir, output_dir)

    all_log_text = all_log.read_text(encoding="utf-8", errors="replace")
    discarded_log_text = discarded_log.read_text(encoding="utf-8", errors="replace")

    # Step 2: Summary statistics.
    logger.info("")
    logger.info("=== Step 2: Summary statistics ===")
    generate_summary(cache_dir, all_log_text, discarded_log_text, output_dir)

    # Step 3: Plots.
    logger.info("")
    logger.info("=== Step 3: Generating plots ===")
    generate_plots(all_log_text, all_log, output_dir)

    # Report output files.
    logger.info("")
    logger.info("======================================================")
    logger.info(" Analysis complete!")
    logger.info(" Output directory: %s", output_dir)
    logger.info("======================================================")
    output_files = sorted(
        f
        for f in output_dir.iterdir()
        if f.is_file() and f.suffix in {".txt", ".log", ".png"}
    )
    for f in output_files:
        size_kb = f.stat().st_size / 1024
        logger.info("  %s (%.1f KB)", f.name, size_kb)
