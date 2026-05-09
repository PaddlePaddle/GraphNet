#!/usr/bin/env python3
import argparse
import os
import sys

from graphsample_insert import insert_one_sample
from init_db import migrate


def insert_from_list(
    list_file,
    model_path_prefix,
    sample_type,
    repo_uid,
    db_path,
    op_names_path_prefix,
    start_order=0,
):
    if not os.path.isfile(list_file):
        print(f"List file not found: {list_file}, skipping")
        return start_order

    with open(list_file) as f:
        paths = [line.strip() for line in f if line.strip()]

    order_value = start_order
    for relative_model_path in paths:
        print(f"insert : {relative_model_path}")
        insert_one_sample(
            model_path_prefix=model_path_prefix,
            relative_model_path=relative_model_path,
            repo_uid=repo_uid,
            sample_type=sample_type,
            order_value=order_value,
            db_path=db_path,
            op_names_path_prefix=op_names_path_prefix,
        )
        order_value += 1

    return order_value


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

    # full_graph
    order_value = insert_from_list(
        list_file=os.path.join(dataset_root, "full_graph.txt"),
        model_path_prefix=os.path.join(dataset_root, "full_graph"),
        sample_type="full_graph",
        repo_uid=repo_uid,
        db_path=db_path,
        op_names_path_prefix="",
        start_order=order_value,
    )

    # typical_graph
    order_value = insert_from_list(
        list_file=os.path.join(dataset_root, "typical_graph.txt"),
        model_path_prefix=os.path.join(dataset_root, "typical_graph"),
        sample_type="typical_graph",
        repo_uid=repo_uid,
        db_path=db_path,
        op_names_path_prefix=op_names_path_prefix,
        start_order=order_value,
    )

    # fusible_graph
    order_value = insert_from_list(
        list_file=os.path.join(dataset_root, "fusible_graph.txt"),
        model_path_prefix=os.path.join(dataset_root, "fusible_graph"),
        sample_type="fusible_graph",
        repo_uid=repo_uid,
        db_path=db_path,
        op_names_path_prefix=op_names_path_prefix,
        start_order=order_value,
    )

    # sole_op_graph
    order_value = insert_from_list(
        list_file=os.path.join(dataset_root, "sole_op_graph.txt"),
        model_path_prefix=os.path.join(dataset_root, "sole_op_graph"),
        sample_type="sole_op_graph",
        repo_uid=repo_uid,
        db_path=db_path,
        op_names_path_prefix=op_names_path_prefix,
        start_order=order_value,
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
