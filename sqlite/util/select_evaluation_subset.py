"""Markov-based greedy evaluation subset selection."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy import sparse


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [0.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def _build_markov_model(
    op_seqs: list[tuple[str, ...]],
) -> tuple[dict[str, int], sparse.csr_matrix, np.ndarray]:
    operators = sorted({op for seq in op_seqs for op in seq})
    op_to_id = {op: idx for idx, op in enumerate(operators)}

    transitions: Counter[tuple[str, str]] = Counter()
    for seq in op_seqs:
        for src, dst in zip(seq[:-1], seq[1:]):
            transitions[(src, dst)] += 1

    rows, cols, data = [], [], []
    for (src, dst), cnt in transitions.items():
        rows.append(op_to_id[src])
        cols.append(op_to_id[dst])
        data.append(cnt)

    size = len(operators)
    count_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    row_sums = np.asarray(count_matrix.sum(axis=1)).ravel()
    return op_to_id, count_matrix, row_sums


def _transition_prob(
    op_to_id: dict[str, int],
    count_matrix: sparse.csr_matrix,
    row_sums: np.ndarray,
    src: str,
    dst: str,
    alpha: float,
) -> float:
    """Laplace-smoothed P(dst | src)."""
    vocab_size = count_matrix.shape[1]
    src_id, dst_id = op_to_id[src], op_to_id[dst]
    count = float(count_matrix[src_id, dst_id])
    denom = float(row_sums[src_id]) + alpha * vocab_size
    return (count + alpha) / denom if denom else 1.0 / vocab_size


def _compute_metrics(
    op_seqs: list[tuple[str, ...]],
    op_to_id: dict[str, int],
    count_matrix: sparse.csr_matrix,
    row_sums: np.ndarray,
    alpha: float,
) -> list[dict]:
    result = []
    for seq in op_seqs:
        edges = list(zip(seq[:-1], seq[1:]))
        if not edges:
            result.append(
                {"rarity_score": 0.0, "unique_edges": frozenset(), "edge_count": 0}
            )
            continue
        neg_log_probs = [
            -math.log(
                _transition_prob(op_to_id, count_matrix, row_sums, src, dst, alpha)
            )
            for src, dst in edges
        ]
        result.append(
            {
                "rarity_score": sum(neg_log_probs) / len(neg_log_probs),
                "unique_edges": frozenset(edges),
                "edge_count": len(edges),
            }
        )
    return result


def _greedy_select(metrics: list[dict], k: int, rarity_weight: float) -> list[int]:
    """Greedily maximise edge coverage, weighted by rarity score."""
    target = min(k, len(metrics))
    rarity_norm = _min_max_normalize([m["rarity_score"] for m in metrics])

    selected: list[int] = []
    selected_set: set[int] = set()
    covered_edges: set[tuple[str, str]] = set()

    while len(selected) < target:
        best_idx, best_key = None, None
        for idx, metric in enumerate(metrics):
            if idx in selected_set:
                continue
            new_edge_count = len(metric["unique_edges"] - covered_edges)
            bonus = rarity_weight * rarity_norm[idx]
            gain = float(new_edge_count) * (1.0 + bonus)
            key = (gain, new_edge_count, bonus, metric["edge_count"])
            if best_key is None or key > best_key:
                best_idx, best_key = idx, key
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        covered_edges.update(metrics[best_idx]["unique_edges"])

    return selected


def select_evaluation_subset(
    op_seqs: list[tuple[str, ...]],
    k: int,
    *,
    smoothing_alpha: float = 1e-3,
    rarity_weight: float = 1,
) -> list[tuple[str, ...]]:
    """Select k sequences from op_seqs using Markov-based greedy coverage.

    Builds a Markov model over all sequences, scores each by rarity
    (mean -log transition prob), then greedily picks sequences that
    maximise new edge coverage weighted by rarity.

    Returns selected sequences in greedy order (most valuable first).
    """
    if k <= 0:
        return []
    seqs = [tuple(s) for s in op_seqs]
    if k >= len(seqs):
        return list(seqs)

    op_to_id, count_matrix, row_sums = _build_markov_model(seqs)
    metrics = _compute_metrics(seqs, op_to_id, count_matrix, row_sums, smoothing_alpha)
    return [seqs[i] for i in _greedy_select(metrics, k, rarity_weight)]
