#!/usr/bin/env python3
"""
Generated from Viba specification:
    select_evaluation_subset := ImportFrom[
        "graph_net.sqlite_util.select_evaluation_subset",
        list[$op_seq list[str]]
        <- list[$op_seq list[str]]
        <- $k int
    ]

    main :=
        void
        <- $op_seq_file ArgParse[FilePathContent[EachLine[JsonStr[$op_seq list[str]]]]]
        <- $k ArgParse[int] # default 200
        # inline
        <- (list[$selected_op_seq list[str]] <- select_evaluation_subset <- list[$op_seq] <- $k)
        <- ($console <- JsonStr[$selected_op_seq])
"""

import argparse
import json
import sys
from typing import List

# Import the target function from the specified module.
# Assumes the module exists and provides a function named 'select_evaluation_subset'.
# The expected signature is: (op_seq_list: List[List[str]], k: int) -> List[List[str]]
# Adjust if the actual signature differs (e.g., curried).
try:
    from graph_net.sqlite_util.select_evaluation_subset import select_evaluation_subset
except ImportError:
    # Fallback for development/testing – replace with actual import.
    def select_evaluation_subset(
        op_seq_list: List[List[str]], k: int
    ) -> List[List[str]]:
        """Dummy implementation: returns first k elements (or fewer)."""
        return op_seq_list[:k]


def parse_op_seq_file(file_path: str) -> List[List[str]]:
    """
    Read a file where each line is a JSON‑encoded list of strings.
    Returns a list of those lists.
    """
    op_seq_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # skip empty lines
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, list) or not all(
                    isinstance(item, str) for item in data
                ):
                    raise ValueError(
                        f"Line {line_num}: expected list of strings, got {type(data)}"
                    )
                op_seq_list.append(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: invalid JSON – {e}")
    return op_seq_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a subset of operation sequences using the specified evaluation strategy."
    )
    parser.add_argument(
        "op_seq_file",
        help="Path to a file where each line is a JSON list of strings (an op_seq).",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=200,
        help="Number of sequences to select (default: 200).",
    )
    args = parser.parse_args()

    # 1. Parse input file
    try:
        op_seq_list = parse_op_seq_file(args.op_seq_file)
    except Exception as e:
        print(f"Error reading op_seq file: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Apply the selection function
    #    (assuming it takes (list_of_sequences, k) – adjust if needed)
    try:
        selected = select_evaluation_subset(op_seq_list, args.k)
    except Exception as e:
        print(f"Error during subset selection: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Output result as JSON to console
    for s in selected:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
