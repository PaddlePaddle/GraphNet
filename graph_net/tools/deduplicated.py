"""Utility for moving duplicate sample models based on their graph hash."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from graph_net.sample_pass.merge_subgraph_sources import (
    MergeSubgraphSources,
)


def collect_graph_hashs(samples_dir: Path) -> Dict[str, List[Path]]:
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    graph_hash2model_paths: Dict[str, List[Path]] = defaultdict(list)
    all_graph_hashs = sorted(samples_dir.rglob("graph_hash.txt"))
    for filepath in all_graph_hashs:
        model_path = filepath.parent
        graph_hash = filepath.read_text(encoding="utf-8").strip()
        graph_hash2model_paths[graph_hash].append(model_path)
    return graph_hash2model_paths


def main(args):
    print(f"Copy samples: {args.samples_dir} -> {args.target_dir}")
    shutil.copytree(args.samples_dir, args.target_dir)
    graph_hash2model_paths = collect_graph_hashs(args.target_dir)
    merge_sources_pass = MergeSubgraphSources()
    num_removed_samples = 0
    for graph_hash, model_paths in graph_hash2model_paths.items():
        if len(model_paths) > 1:
            # Keep the first sample and merge sources from all duplicates
            target_path = model_paths[0]
            duplicate_paths = model_paths[1:]
            merge_sources_pass.merge_sources_for_deduplication(
                target_path, duplicate_paths
            )
            # Remove the duplicate samples
            for dup_path in duplicate_paths:
                print(f"Remove  {dup_path}")
                shutil.rmtree(dup_path)
                num_removed_samples += 1
    print(
        f"Totally {len(graph_hash2model_paths)} different graph_hashs, {num_removed_samples} samples are removed."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-dir",
        type=Path,
        required=True,
        default=None,
        help="Root directory containing sample models.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory where duplicate models will be moved to.",
    )
    args = parser.parse_args()
    main(args)
