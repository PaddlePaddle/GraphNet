#!/usr/bin/env python3
"""
select_representative_sample_uids.py

Reads a TSV file of (sample_uid, op_seq) pairs and a file listing selected op_seqs.
For each selected op_seq, picks the sample_uid with the maximum string length from its group,
and outputs them in the same order as the selected op_seqs.
"""

import argparse
import sys
from collections import defaultdict


def read_tsv_pairs(file_path):
    """
    Read a tab-separated file where each line contains sample_uid and op_seq.
    Returns a list of (sample_uid, op_seq) tuples.
    """
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                print(
                    f"Warning: {file_path}:{line_num} invalid format, skipped: {line}",
                    file=sys.stderr,
                )
                continue
            sample_uid, op_seq = parts
            pairs.append((sample_uid, op_seq))
    return pairs


def read_op_seq_list(file_path):
    """
    Read a file with one op_seq per line. Returns a list preserving order.
    """
    op_seqs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                op_seqs.append(line)
    return op_seqs


def get_max_len_string(strings):
    """
    Return the string with the maximum length from a list.
    If multiple strings have the same max length, return the first encountered.
    """
    if not strings:
        return None
    # max with key returns the first element in case of ties (Python's max is stable)
    return max(strings, key=lambda s: len(s))


def main():
    parser = argparse.ArgumentParser(
        description="Select representative sample UIDs by max string length per op_seq group."
    )
    parser.add_argument(
        "pairs_file", help="TSV file with two columns: sample_uid and op_seq"
    )
    parser.add_argument(
        "opseq_file", help="File with one op_seq per line (order matters)"
    )
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    args = parser.parse_args()

    # 1. Load input data
    pairs = read_tsv_pairs(args.pairs_file)
    selected_op_seqs = read_op_seq_list(args.opseq_file)

    # 2. Group sample_uids by op_seq
    groups = defaultdict(list)
    for sample_uid, op_seq in pairs:
        groups[op_seq].append(sample_uid)

    # 3. For each op_seq, find the longest sample_uid
    op_seq_to_max_uid = {}
    for op_seq, uid_list in groups.items():
        max_uid = get_max_len_string(uid_list)
        if max_uid is not None:
            op_seq_to_max_uid[op_seq] = max_uid

    # 4. Collect results in the order of selected_op_seqs
    result_uids = []
    for op_seq in selected_op_seqs:
        if op_seq in op_seq_to_max_uid:
            result_uids.append(op_seq_to_max_uid[op_seq])

    # 5. Write output
    out_fh = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for uid in result_uids:
            print(uid, file=out_fh)
    finally:
        if args.output:
            out_fh.close()


if __name__ == "__main__":
    main()
