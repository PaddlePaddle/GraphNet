"""Step 4: Extract autotuning-selected triton kernels and corresponding PTX."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Compiled regex that replaces the original perl one-liner:
#
#   perl -0777 -ne '
#       while (
#         /async_compile\.triton\(\x27([^\x27]+)\x27,\s*\x27\x27\x27(.*?)\x27\x27\x27/gs
#       ) {
#           print "===KERNEL_NAME===$1\n$2\n===KERNEL_END===\n";
#       }'
#
# Captures: group(1) = kernel name, group(2) = kernel source code.
_TRITON_KERNEL_PATTERN = re.compile(
    r"async_compile\.triton\('([^']+)',\s*'''(.*?)'''",
    re.DOTALL,
)
_KERNEL_PATH_PATTERN = re.compile(r"^# kernel path:\s*(.+)$", re.MULTILINE)
_MAKE_RANGE_END_PATTERN = re.compile(r"tt\.make_range\s*\{end = (\d+)")
_BLOCK_KEY_ORDER = ("XBLOCK", "YBLOCK", "ZBLOCK", "RBLOCK")


def _collect_best_configs(
    sample_cache_dir: Path,
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Gather autotuning-selected configs from a sample directory.

    TorchInductor writes ``.best_config`` JSON files (one per autotuned kernel)
    in 2-char prefix subdirectories of the sample cache.  Older versions expose
    a ``triton_cache_hash`` field identifying the winning candidate under
    ``triton/0/``.  Newer TorchInductor/Triton combinations may leave that field
    as ``null``; in that case the prefix directory from ``# kernel path:`` in
    ``output_code.py`` is used to find the matching best-config and compare the
    recorded tuning parameters against candidate metadata.

    This function is called once per sample and the result is reused for every
    kernel in that sample.
    """
    hashes: set[str] = set()
    by_prefix: dict[str, dict[str, Any]] = {}
    for bc_path in sample_cache_dir.rglob("*.best_config"):
        try:
            data = json.loads(bc_path.read_text(encoding="utf-8"))
            cache_hash = data.get("triton_cache_hash")
            if cache_hash:
                hashes.add(cache_hash)
            by_prefix[bc_path.parent.name] = data
        except (OSError, json.JSONDecodeError):
            logger.debug("Skipping malformed .best_config: %s", bc_path)
    return hashes, by_prefix


def _extract_block_values_from_source(source_path: Path) -> list[int]:
    """Extract Triton block dimensions from compact metadata or ``*.source``."""
    meta_path = source_path.with_suffix(".pruned_meta.json")
    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            block_values = data.get("block_values")
            if isinstance(block_values, list):
                return [int(v) for v in block_values]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return []

    try:
        content = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [int(m) for m in _MAKE_RANGE_END_PATTERN.findall(content)]


def _candidate_matches_best_config(ptx_path: Path, best_config: dict[str, Any]) -> bool:
    """Return True when a PTX candidate matches a ``.best_config`` record."""
    json_path = ptx_path.with_suffix(".json")
    if json_path.is_file():
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    else:
        metadata = {}

    for key in ("num_warps", "num_stages"):
        if key in best_config and key in metadata and metadata[key] != best_config[key]:
            return False

    block_values = _extract_block_values_from_source(ptx_path.with_suffix(".source"))
    best_block_keys = [key for key in _BLOCK_KEY_ORDER if key in best_config]
    if best_block_keys:
        if len(block_values) < len(best_block_keys):
            return False
        for idx, key in enumerate(best_block_keys):
            if block_values[idx] != best_config[key]:
                return False

    # If only some metadata files survived, accept the candidate only when at
    # least one meaningful field was available and matched.
    return bool(
        any(key in metadata for key in ("num_warps", "num_stages")) or block_values
    )


def _find_best_ptx(
    sample_cache_dir: Path,
    kernel_name: str,
    best_hashes: set[str],
    best_config: dict[str, Any] | None = None,
) -> str | None:
    """Locate the corresponding PTX for a given kernel via autotuning results.

    The inductor cache compiles each triton kernel into one or more candidate
    configurations under ``triton/0/{HASH}/``.  When autotuning runs, multiple
    candidate directories exist for the same kernel and a ``.best_config`` file
    records the winning ``triton_cache_hash``.

    Resolution strategy:
      - 0 candidates → return ``None`` (no PTX compiled for this kernel).
      - 1 candidate  → return its PTX (no disambiguation needed).
      - N candidates → first intersect directory names with *best_hashes*.
      - If no hash is available, compare candidate ``*.json`` / ``*.source``
        metadata against the kernel's ``.best_config``.
    """
    triton_base = sample_cache_dir / "triton" / "0"
    if not triton_base.is_dir():
        return None

    # Collect all triton/0/{hash}/ dirs that contain this kernel's PTX.
    ptx_filename = f"{kernel_name}.ptx"
    candidates: list[Path] = [
        ptx_file
        for hash_dir in triton_base.iterdir()
        if hash_dir.is_dir()
        for ptx_file in [hash_dir / ptx_filename]
        if ptx_file.is_file()
    ]

    if not candidates:
        logger.debug(
            "No PTX found for kernel %s in %s", kernel_name, sample_cache_dir.name
        )
        return None

    if len(candidates) == 1:
        try:
            return candidates[0].read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.warning("Cannot read PTX file: %s", candidates[0])
            return None

    # Multiple candidates: pick the one whose parent dir matches a best_config hash.
    for ptx_path in candidates:
        if ptx_path.parent.name in best_hashes:
            try:
                return ptx_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.warning("Cannot read PTX file: %s", ptx_path)
                return None

    if best_config:
        metadata_matches = [
            ptx_path
            for ptx_path in candidates
            if _candidate_matches_best_config(ptx_path, best_config)
        ]
        if len(metadata_matches) == 1:
            try:
                return metadata_matches[0].read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.warning("Cannot read PTX file: %s", metadata_matches[0])
                return None
        if len(metadata_matches) > 1:
            logger.warning(
                "Multiple metadata-matched PTX candidates for %s in %s",
                kernel_name,
                sample_cache_dir.name,
            )
            return None

    # Fallback: no .best_config match.
    logger.warning(
        "Multiple PTX candidates for %s but no .best_config match in %s",
        kernel_name,
        sample_cache_dir.name,
    )
    return None


def extract_kernels_from_file(
    output_code_path: Path,
) -> list[tuple[str, str, str | None]]:
    """Parse an ``output_code.py`` and return ``(name, source, prefix)`` tuples.

    The file is read entirely into memory (``output_code.py`` files produced by
    TorchInductor are typically well under 1 MB).  Returns an empty list if the
    file cannot be read.
    """
    try:
        content = output_code_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("Cannot read output_code.py: %s", output_code_path)
        return []

    kernels: list[tuple[str, str, str | None]] = []
    kernel_path_matches = list(_KERNEL_PATH_PATTERN.finditer(content))
    for match in _TRITON_KERNEL_PATTERN.finditer(content):
        prefix: str | None = None
        preceding_paths = [m for m in kernel_path_matches if m.start() < match.start()]
        if preceding_paths:
            kernel_path = preceding_paths[-1].group(1).strip()
            prefix = Path(kernel_path).parent.name or None
        kernels.append((match.group(1), match.group(2), prefix))
    return kernels


def extract_triton_kernels(
    cache_dir: Path,
    output_dir: Path,
) -> tuple[int, int, int, int, int]:
    """Walk kept samples and extract selected triton kernels with PTX.

    For every kept sample that contains ``original_graph/graph_hash.txt``:

    1. Copy ``original_graph/model.py`` into the output.
    2. Parse every ``output_code.py`` found in the sample tree.
    3. Write each extracted kernel to ``triton_kernel/{name}.py``.
    4. Locate the corresponding PTX for each kernel and write it to
       ``ptx/{name}.ptx``.

    The output uses an atomic ``.tmp`` + ``rename`` pattern so that an
    interrupted run never leaves a half-written sample directory.

    Returns:
        ``(processed_files, total_kernels, total_ptx, extracted_samples, skip_count)``
    """
    kept_dir = cache_dir / "kept"
    if not kept_dir.is_dir():
        logger.error("Kept directory does not exist: %s", kept_dir)
        return 0, 0, 0, 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up stale .tmp directories from a previous interrupted run.
    for stale in output_dir.iterdir():
        if stale.is_dir() and stale.name.endswith(".tmp"):
            shutil.rmtree(stale, ignore_errors=True)

    # Collect eligible samples (must contain original_graph/graph_hash.txt).
    eligible: list[Path] = [
        d
        for d in sorted(kept_dir.iterdir())
        if d.is_dir() and (d / "original_graph" / "graph_hash.txt").is_file()
    ]
    total = len(eligible)

    processed_files = 0
    total_kernels = 0
    total_ptx = 0
    extracted_samples = 0
    skip_count = 0

    for idx, sample_cache_dir in enumerate(eligible, 1):
        sample_name = sample_cache_dir.name
        dest_sample_dir = output_dir / sample_name

        # Resume: skip if the final output already exists.
        if dest_sample_dir.exists():
            skip_count += 1
            continue

        logger.info("[%d/%d] Extracting: %s", idx, total, sample_name)

        # Write to a temporary directory; rename atomically on success.
        tmp_dir = dest_sample_dir.with_name(f"{sample_name}.tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        # Copy original model source when available.
        model_src = sample_cache_dir / "original_graph" / "model.py"
        if model_src.is_file():
            og_dir = tmp_dir / "original_graph"
            og_dir.mkdir()
            shutil.copy2(str(model_src), str(og_dir / "model.py"))

        # Pre-collect autotuning best-config data once per sample.
        best_hashes, best_configs_by_prefix = _collect_best_configs(sample_cache_dir)

        # Track kernel names already written for this sample to detect
        # duplicates across multiple output_code.py files.
        seen_kernels: set[str] = set()

        # Find and process all output_code.py files within the sample.
        for output_code_path in sorted(sample_cache_dir.rglob("output_code.py")):
            processed_files += 1
            kernels = extract_kernels_from_file(output_code_path)
            if not kernels:
                continue

            triton_dir = tmp_dir / "triton_kernel"
            triton_dir.mkdir(exist_ok=True)

            for name, source, kernel_prefix in kernels:
                if name in seen_kernels:
                    logger.debug(
                        "Duplicate kernel %s in %s, skipping", name, sample_name
                    )
                    continue
                seen_kernels.add(name)
                # Strip trailing whitespace then add exactly one newline,
                # matching the bash `printf '%s\n'` semantics.
                (triton_dir / f"{name}.py").write_text(
                    source.rstrip() + "\n", encoding="utf-8"
                )
                total_kernels += 1

                # Locate and write the corresponding PTX for this kernel.
                best_config = (
                    best_configs_by_prefix.get(kernel_prefix)
                    if kernel_prefix is not None
                    else None
                )
                ptx_content = _find_best_ptx(
                    sample_cache_dir,
                    name,
                    best_hashes,
                    best_config=best_config,
                )
                if ptx_content is not None:
                    ptx_dir = tmp_dir / "ptx"
                    ptx_dir.mkdir(exist_ok=True)
                    (ptx_dir / f"{name}.ptx").write_text(ptx_content, encoding="utf-8")
                    total_ptx += 1

        # Atomic completion: rename .tmp → final (same filesystem guarantees
        # a single rename(2) syscall).
        tmp_dir.rename(dest_sample_dir)
        extracted_samples += 1

    logger.info(
        "Extraction: %d files, %d kernels, %d ptx, %d samples, %d skipped (total: %d)",
        processed_files,
        total_kernels,
        total_ptx,
        extracted_samples,
        skip_count,
        total,
    )
    logger.info("Output:     %s", output_dir)
    return processed_files, total_kernels, total_ptx, extracted_samples, skip_count
