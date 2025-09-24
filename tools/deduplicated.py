#!/usr/bin/env python3
"""Utility for moving duplicate sample models based on their graph hash."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def collect_graph_hash_files(samples_dir: Path) -> List[Path]:
    """Return all ``graph_hash.txt`` files under ``samples_dir`` in sorted order."""

    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    return sorted(samples_dir.rglob("graph_hash.txt"))


def build_hash_to_models(graph_hash_files: Iterable[Path]) -> Dict[str, List[Path]]:
    """Map each graph hash value to the model directories that contain it."""

    hash_to_models: Dict[str, List[Path]] = defaultdict(list)

    for hash_file in graph_hash_files:
        model_dir = hash_file.parent
        hash_value = hash_file.read_text(encoding="utf-8").strip()
        hash_to_models[hash_value].append(model_dir)

    return hash_to_models


def compute_duplicates(hash_to_models: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
    """Filter the mapping to only hash values that appear more than once."""

    return {
        hash_value: dirs for hash_value, dirs in hash_to_models.items() if len(dirs) > 1
    }


def move_duplicates(
    duplicates: Dict[str, List[Path]],
    samples_dir: Path,
    redundant_dir: Path,
    dry_run: bool = False,
) -> None:
    """Move duplicate model directories under ``redundant_dir``.

    Directories are moved while preserving their relative path with respect to
    ``samples_dir``.
    """

    redundant_dir.mkdir(parents=True, exist_ok=True)

    for hash_value, model_dirs in sorted(duplicates.items()):
        # Keep the first occurrence inside ``samples`` and move the rest.
        keep_dir, *dupe_dirs = model_dirs
        for dupe_dir in dupe_dirs:
            relative_path = dupe_dir.relative_to(samples_dir)
            destination = redundant_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if dry_run:
                print(f"[DRY-RUN] Would move {dupe_dir} -> {destination}")
                continue

            if destination.exists():
                raise FileExistsError(f"Destination already exists: {destination}")

            print(f"Moving duplicate hash {hash_value}: {dupe_dir} -> {destination}")
            shutil.move(str(dupe_dir), str(destination))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "samples",
        help="Root directory containing sample models.",
    )
    parser.add_argument(
        "--redundant-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "redundant",
        help="Directory where duplicate models will be moved.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report duplicate moves without performing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_hash_files = collect_graph_hash_files(args.samples_dir)
    hash_to_models = build_hash_to_models(graph_hash_files)
    duplicates = compute_duplicates(hash_to_models)

    if not duplicates:
        print("No duplicate graph hashes found.")
        return

    move_duplicates(
        duplicates, args.samples_dir, args.redundant_dir, dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
