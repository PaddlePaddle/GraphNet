# vectorize_op_seq.py
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass


@dataclass
class VectorizationResult:
    """Result of vectorizing a list of operation sequences."""

    samples: List[List[int]]  # list of feature vectors, one per input sequence
    embedding: List[
        Tuple[str, ...]
    ]  # list of unique sub‑operation sequences (features)


def vectorize_op_seq(
    input_op_seqs: List[List[str]], sub_op_seq_len_limit: int = 6
) -> VectorizationResult:
    """
    Convert a list of operation sequences into a bag‑of‑subsequences representation.

    For each input sequence, all contiguous subsequences of length 1 up to
    `sub_op_seq_len_limit` are extracted. A global vocabulary (embedding) of all
    distinct subsequences is built, and each sequence is represented as a count
    vector over that vocabulary.

    Args:
        input_op_seqs: List of operation sequences, each being a list of strings.
        sub_op_seq_len_limit: Maximum length of a subsequence (default 6).

    Returns:
        VectorizationResult containing:
            - samples: list of integer count vectors (one per input sequence)
            - embedding: list of distinct subsequences (as tuples of strings)
    """
    # ------------------------------------------------------------------
    # Step 1: Build the set of all possible subsequences (embedding_set)
    # ------------------------------------------------------------------
    embedding_set: Set[Tuple[str, ...]] = set()

    for seq in input_op_seqs:
        seq_len = len(seq)
        # Iterate over all possible start positions
        for start in range(seq_len):
            # Iterate over all possible lengths (1 .. sub_op_seq_len_limit)
            max_len = min(sub_op_seq_len_limit, seq_len - start)
            for length in range(1, max_len + 1):
                subseq = tuple(seq[start : start + length])
                embedding_set.add(subseq)

    # ------------------------------------------------------------------
    # Step 2: Create a sorted embedding list (deterministic order)
    # ------------------------------------------------------------------
    embedding = sorted(embedding_set)  # sort by tuple content
    # Map each subsequence to its index in the embedding
    embedding_map: Dict[Tuple[str, ...], int] = {
        subseq: idx for idx, subseq in enumerate(embedding)
    }

    # ------------------------------------------------------------------
    # Step 3: Build feature vectors for each input sequence
    # ------------------------------------------------------------------
    samples: List[List[int]] = []

    for seq in input_op_seqs:
        # Initialize count vector with zeros
        counts = [0] * len(embedding)
        seq_len = len(seq)

        for start in range(seq_len):
            max_len = min(sub_op_seq_len_limit, seq_len - start)
            for length in range(1, max_len + 1):
                subseq = tuple(seq[start : start + length])
                idx = embedding_map[subseq]
                counts[idx] += 1

        samples.append(counts)

    return VectorizationResult(samples=samples, embedding=embedding)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Test cases
    # ------------------------------------------------------------------

    # Test 1: empty input
    result = vectorize_op_seq([])
    assert result.samples == []
    assert result.embedding == []
    print("Test 1 passed (empty input)")

    # Test 2: single sequence, subsequence length limit = 2
    seq = ["a", "b", "c"]
    result = vectorize_op_seq([seq], sub_op_seq_len_limit=2)
    # Expected embedding (sorted): ('a',), ('a','b'), ('b',), ('b','c'), ('c',)
    expected_embedding = [("a",), ("a", "b"), ("b",), ("b", "c"), ("c",)]
    assert result.embedding == expected_embedding
    # Feature vector: each subsequence appears exactly once
    assert result.samples[0] == [1, 1, 1, 1, 1]
    print("Test 2 passed (single seq, limit=2)")

    # Test 3: multiple sequences, limit = 3
    seqs = [["x", "y"], ["x", "y", "z"], ["a", "b"]]
    result = vectorize_op_seq(seqs, sub_op_seq_len_limit=3)
    # Expected embedding (sorted): all distinct subsequences up to length 3
    # Generate manually:
    all_subseqs = set()
    for s in seqs:
        for start in range(len(s)):
            for sub_seq_len in range(1, min(3, len(s) - start) + 1):
                all_subseqs.add(tuple(s[start : start + sub_seq_len]))
    expected_embedding = sorted(all_subseqs)
    assert result.embedding == expected_embedding
    # Check sample dimensions
    assert len(result.samples) == 3
    assert len(result.samples[0]) == len(expected_embedding)
    # Spot check: first sequence ["x","y"] should have counts:
    # ('x'):1, ('x','y'):1, ('y'):1, others 0
    idx_x = result.embedding.index(("x",))
    idx_xy = result.embedding.index(("x", "y"))
    idx_y = result.embedding.index(("y",))
    assert result.samples[0][idx_x] == 1
    assert result.samples[0][idx_xy] == 1
    assert result.samples[0][idx_y] == 1
    print("Test 3 passed (multiple sequences)")

    # Test 4: limit larger than sequence length (should only go to seq length)
    seq = ["p", "q"]
    result = vectorize_op_seq([seq], sub_op_seq_len_limit=10)
    assert result.embedding == [("p",), ("p", "q"), ("q",)]
    assert result.samples[0] == [1, 1, 1]
    print("Test 4 passed (limit > seq length)")

    # Test 5: repeated subsequences (counting)
    seq = ["a", "a", "a"]  # all elements same
    result = vectorize_op_seq([seq], sub_op_seq_len_limit=2)
    # embedding: ('a',), ('a','a')
    # subsequences:
    # start 0: ('a') len1, ('a','a') len2
    # start 1: ('a') len1, ('a','a') len2
    # start 2: ('a') len1
    # counts: ('a') = 3, ('a','a') = 2
    assert result.samples[0] == [3, 2]
    print("Test 5 passed (repeated subsequences)")

    # Test 6: different sequences produce consistent embedding
    seqs = [["a", "b"], ["b", "a"]]
    result = vectorize_op_seq(seqs, sub_op_seq_len_limit=2)
    # embedding sorted: ('a',), ('a','b'), ('b',), ('b','a')
    assert result.embedding == [("a",), ("a", "b"), ("b",), ("b", "a")]
    # first seq ["a","b"]: counts: ('a'):1, ('a','b'):1, ('b'):1
    assert result.samples[0] == [1, 1, 1, 0]
    # second seq ["b","a"]: ('b'):1, ('b','a'):1, ('a'):1
    assert result.samples[1] == [1, 0, 1, 1]
    print("Test 6 passed (different orders)")

    print("All tests passed.")
