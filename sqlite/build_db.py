#!/usr/bin/env python3
import argparse
import os
import sys

from graphsample_insert import insert_one_sample
from init_db import migrate


def collect_sample_paths(model_path_prefix):
    """Collect relative paths of directories containing model.py under model_path_prefix."""
    sample_paths = []
    for dirpath, _, filenames in os.walk(model_path_prefix):
        if "model.py" in filenames:
            rel = os.path.relpath(dirpath, model_path_prefix)
            sample_paths.append(rel)
    sample_paths.sort()
    return sample_paths


def insert_from_list(
    list_file_path,
    model_path_prefix,
    sample_type,
    repo_uid,
    db_path,
    op_names_path_prefix,
    start_order=0,
):
    if os.path.isfile(list_file_path):
        with open(list_file_path) as f:
            sample_paths = [line.strip() for line in f if line.strip()]
        sample_paths.sort()
    else:
        print(
            f"List file not found: {list_file_path}, collecting from {model_path_prefix}"
        )
        sample_paths = collect_sample_paths(model_path_prefix)

    total = len(sample_paths)
    order_value = start_order
    for relative_model_path in sample_paths:
        print(f"insert : {relative_model_path}")
        successed = insert_one_sample(
            model_path_prefix=model_path_prefix,
            relative_model_path=relative_model_path,
            repo_uid=repo_uid,
            sample_type=sample_type,
            order_value=order_value,
            db_path=db_path,
            op_names_path_prefix=op_names_path_prefix,
        )
        if successed:
            order_value += 1

    return order_value, total


def main(args):
    dataset_root = args.dataset_root.strip()
    db_path = args.db_path.strip()
    repo_uid = args.repo_uid.strip()
    op_names_path_prefix = args.op_names_path_prefix.strip()

    if not os.path.exists(db_path):
        migrate(db_path)
    else:
        print(f"Fail ! Path is not a file: {db_path}")
        sys.exit(1)

    order_value = 0

    sample_types = [
        ("full_graph", ""),
        ("typical_graph", op_names_path_prefix),
        ("fusible_graph", op_names_path_prefix),
        ("sole_op_graph", op_names_path_prefix),
    ]
    for sample_type, op_prefix in sample_types:
        order_start = order_value
        order_value, total = insert_from_list(
            list_file_path=os.path.join(dataset_root, f"{sample_type}.txt"),
            model_path_prefix=os.path.join(dataset_root, sample_type),
            sample_type=sample_type,
            repo_uid=repo_uid,
            db_path=db_path,
            op_names_path_prefix=op_prefix,
            start_order=order_value,
        )
        num_success = order_value - order_start
        print(
            f"[{sample_type}] total={total}, success={num_success}, "
            f"fail={total - num_success}, order=[{order_start}, {order_value})"
        )

    print("all done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch insert graph samples from list files"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Database file path",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--repo_uid",
        type=str,
        default="hf_torch_samples",
        help="Repository uid",
    )
    parser.add_argument(
        "--op_names_path_prefix",
        type=str,
        required=True,
        help="Path prefix of op names files",
    )
    args = parser.parse_args()
    main(args)
