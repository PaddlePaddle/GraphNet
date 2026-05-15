#!/usr/bin/env python3
"""Backward kernel dedup analysis tool.

Usage:
    python tools/backward_kernel_dedup.py \
        --backward-dir /tmp/bw_results/backward_graphs \
        --tag typical_backward \
        --output /tmp/bw_dedup.json
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


def compile_and_extract_kernels(model_path, device="cuda"):
    """Compile a backward model with inductor and extract kernel sources."""
    kernels = []
    try:
        module_name = Path(model_path).name
        # Import the model dynamically
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(model_path, "model.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        model = mod.GraphModule().to(device)

        # We need dummy inputs; try to load from weight_meta / input_meta if available
        inputs = []
        for meta_file in ["input_meta.py", "weight_meta.py"]:
            meta_path = os.path.join(model_path, meta_file)
            if not os.path.exists(meta_path):
                continue
            spec = importlib.util.spec_from_file_location(
                f"{module_name}_{meta_file}", meta_path
            )
            meta_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(meta_mod)
            for attr_name in dir(meta_mod):
                attr = getattr(meta_mod, attr_name)
                if (
                    isinstance(attr, type)
                    and hasattr(attr, "shape")
                    and hasattr(attr, "dtype")
                ):
                    shape = attr.shape
                    dtype_str = attr.dtype.replace("torch.", "")
                    dtype_map = {
                        "float32": torch.float32,
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                        "int64": torch.int64,
                        "int32": torch.int32,
                        "bool": torch.bool,
                    }
                    dtype = dtype_map.get(dtype_str, torch.float32)
                    if "int" in dtype_str or "bool" in dtype_str:
                        t = torch.zeros(shape, dtype=dtype, device=device)
                    else:
                        t = torch.randn(shape, dtype=dtype, device=device)
                    inputs.append(t)

        if not inputs:
            return []

        compiled = torch.compile(model, backend="inductor")
        _ = compiled(*inputs)
        if "cuda" in device:
            torch.cuda.synchronize()

        # Try to read inductor generated code from cache
        cache_dir = os.path.expanduser("~/.torchinductor")
        if os.path.exists(cache_dir):
            for root, _, files in os.walk(cache_dir):
                for f in files:
                    if f.endswith(".py"):
                        with open(os.path.join(root, f), "r", encoding="utf-8") as fp:
                            content = fp.read()
                            if "triton" in content.lower():
                                kernels.append(content)
    except Exception as e:
        print(f"Error extracting kernels from {model_path}: {e}")
    return kernels


def hash_kernel(kernel_code):
    """Compute a simple hash for a kernel source."""
    # Normalize by removing comments and extra whitespace
    lines = [
        line.strip()
        for line in kernel_code.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    normalized = "\n".join(lines)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def analyze_kernels(backward_dir, tag):
    """Analyze kernel dedup for all backward graphs under backward_dir."""
    samples = []
    for dirpath, _, filenames in os.walk(backward_dir):
        if "model.py" in filenames:
            samples.append(dirpath)

    all_hashes = []
    sample_kernels = []

    for sample_path in samples:
        print(f"Processing {sample_path} ...")
        kernels = compile_and_extract_kernels(sample_path)
        hashes = [hash_kernel(k) for k in kernels]
        all_hashes.extend(hashes)
        sample_kernels.append(
            {
                "path": sample_path,
                "kernel_count": len(kernels),
                "hashes": hashes,
            }
        )

    total = len(all_hashes)
    unique = len(set(all_hashes))
    dedup_rate = (1 - unique / total) * 100 if total > 0 else 0

    summary = {
        "tag": tag,
        "total_samples": len(samples),
        "total_kernel_instances": total,
        "unique_kernels": unique,
        "dedup_rate_percent": round(dedup_rate, 2),
        "avg_kernels_per_graph": round(total / len(samples), 2) if samples else 0,
        "per_sample": sample_kernels,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Backward kernel dedup analysis.")
    parser.add_argument(
        "--backward-dir",
        type=str,
        required=True,
        help="Directory containing backward graph subdirectories",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="backward",
        help="Tag for the analysis (e.g., typical_backward, fusible_backward)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path for dedup results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for compilation",
    )
    args = parser.parse_args()

    summary = analyze_kernels(args.backward_dir, args.tag)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Dedup Summary ({args.tag}) ===")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total kernel instances: {summary['total_kernel_instances']}")
    print(f"Unique kernels: {summary['unique_kernels']}")
    print(f"Dedup rate: {summary['dedup_rate_percent']}%")
    print(f"Avg kernels/graph: {summary['avg_kernels_per_graph']}")
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
