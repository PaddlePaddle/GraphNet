#!/bin/bash
set -euo pipefail

# Thin launcher for the triton kernel extraction pipeline.
#
# This script sets machine-specific paths and invokes the Python pipeline
# once per dataset category, using pre-computed allow-list files.
#
# Usage:
#   bash tools/extract_triton_kernels.sh [gpu_ids]
#
# Args:
#   gpu_ids  (optional): comma-separated GPU IDs, e.g. "0,2,5,7"
#
# Examples:
#   bash tools/extract_triton_kernels.sh              # auto-detect GPUs
#   bash tools/extract_triton_kernels.sh 0,2,5,7      # specified GPUs

# ============================================================
# Arguments
# ============================================================

GPU_ARG="${1:-}"

# ============================================================
# Machine-specific path configuration
#
# Edit the variables below to match your local environment.
# ============================================================

# Root directory containing graph data (base for resolving paths in allow-lists).
GRAPH_DIR="/path/to/input_graph_data"

# Root directory for pipeline output (cache + extracted kernels).
OUTPUT_BASE="/path/to/dataset_output"

# Dataset categories to process.
CATEGORIES=(sole_op_subgraphs fusible_subgraphs typical_subgraphs)

# Allow-list files (one per category, same order as CATEGORIES).
ALLOW_LISTS=(
    "${OUTPUT_BASE}/hf_sole_op_samples_v2_all_expanded.txt"
    "${OUTPUT_BASE}/hf_fusible_samples_v2_all_expanded.txt"
    "${OUTPUT_BASE}/hf_typical_samples_v2_all_expanded.txt"
)

# ============================================================
# Environment setup
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Validate that placeholder paths have been configured.
if [ "$GRAPH_DIR" = "/path/to/input_graph_data" ] || [ "$OUTPUT_BASE" = "/path/to/dataset_output" ]; then
    echo "ERROR: Edit GRAPH_DIR and OUTPUT_BASE in this script before running." >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ============================================================
# Build GPU args
# ============================================================

GPU_ARGS=()
if [ -n "$GPU_ARG" ]; then
    IFS=',' read -ra GPU_IDS <<< "$GPU_ARG"
    GPU_ARGS=(--gpu-ids "${GPU_IDS[@]}")
fi

# ============================================================
# Run pipeline for each category
# ============================================================

for i in "${!CATEGORIES[@]}"; do
    CATEGORY="${CATEGORIES[$i]}"

    python3 -m tools.triton_kernel_extractor extract \
        --allow-list "${ALLOW_LISTS[$i]}" \
        --graph-dir "$GRAPH_DIR" \
        --output-dir "${OUTPUT_BASE}/${CATEGORY}_inductor_dump" \
        --max-autotune-no-cudagraphs \
        --enable-cache-analysis \
        "${GPU_ARGS[@]}"
done

echo "All categories processed."
