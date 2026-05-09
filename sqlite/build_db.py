#!/usr/bin/env python3
import argparse
import os
import sys

import graph_net

from graphsample_insert import insert_one_sample


GRAPH_NET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(graph_net.__file__)))


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


def main():
    parser = argparse.ArgumentParser(
        description="Batch insert graph samples from list files"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help="Database file path (default: GRAPH_NET_ROOT/sqlite/GraphNet.db)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Dataset root directory (default: GRAPH_NET_ROOT/20260317)",
    )
    parser.add_argument(
        "--repo_uid",
        type=str,
        default="hf_torch_samples",
        help="Repository uid",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root or os.path.join(GRAPH_NET_ROOT, "20260317")
    db_path = args.db_path or os.path.join(GRAPH_NET_ROOT, "sqlite", "GraphNet.db")
    repo_uid = args.repo_uid.strip()

    if not os.path.isfile(db_path):
        print(f"Fail ! No Database ! : {db_path}")
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
        op_names_path_prefix=os.path.join(dataset_root, "03_sample_op_names"),
        start_order=order_value,
    )

    # fusible_graph
    order_value = insert_from_list(
        list_file=os.path.join(dataset_root, "fusible_graph.txt"),
        model_path_prefix=os.path.join(dataset_root, "fusible_graph"),
        sample_type="fusible_graph",
        repo_uid=repo_uid,
        db_path=db_path,
        op_names_path_prefix=os.path.join(dataset_root, "03_sample_op_names"),
        start_order=order_value,
    )

    # sole_op_graph
    order_value = insert_from_list(
        list_file=os.path.join(dataset_root, "sole_op_graph.txt"),
        model_path_prefix=os.path.join(dataset_root, "sole_op_graph"),
        sample_type="sole_op_graph",
        repo_uid=repo_uid,
        db_path=db_path,
        op_names_path_prefix=os.path.join(dataset_root, "03_sample_op_names"),
        start_order=order_value,
    )

    print("all done")


if __name__ == "__main__":
    main()
