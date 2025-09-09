from . import utils
import argparse
import importlib.util
import inspect
from pathlib import Path
from typing import Type, Any
import sys
import os
import os.path
from dataclasses import dataclass
from contextlib import contextmanager
import time
import glob


def get_recursively_model_pathes(root_dir):
    for sub_dir in _get_recursively_model_pathes(root_dir):
        yield os.path.realpath(sub_dir)


def _get_recursively_model_pathes(root_dir):
    if is_single_model_dir(root_dir):
        yield root_dir
        return
    for sub_dir in get_immediate_subdirectory_paths(root_dir):
        if is_single_model_dir(sub_dir):
            yield sub_dir
        else:
            yield from get_recursively_model_pathes(sub_dir)


def get_immediate_subdirectory_paths(parent_dir):
    return [
        sub_dir
        for name in os.listdir(parent_dir)
        for sub_dir in [os.path.join(parent_dir, name)]
        if os.path.isdir(sub_dir)
    ]


def is_single_model_dir(model_dir):
    return os.path.isfile(f"{model_dir}/graph_net.json")


def main(args):
    assert os.path.isdir(
        args.graph_net_samples_path
    ), f"args.graph_net_samples_path ({args.graph_net_samples_path}) is not a directory!"
    find_redundant = False
    graph_hash2graph_net_model_path = {}
    for model_path in get_recursively_model_pathes(args.graph_net_samples_path):
        graph_hash_path = f"{model_path}/graph_hash.txt"
        if os.path.isfile(graph_hash_path):
            graph_hash = open(graph_hash_path).read()
            if graph_hash not in graph_hash2graph_net_model_path.keys():
                graph_hash2graph_net_model_path[graph_hash] = [graph_hash_path]
            else:
                find_redundant = True
                graph_hash2graph_net_model_path[graph_hash].append(graph_hash_path)
    print(
        f"Totally {len(graph_hash2graph_net_model_path)} unique samples under {args.graph_net_samples_path}."
    )
    for graph_hash, graph_paths in graph_hash2graph_net_model_path.items():
        if len(graph_paths) > 1:
            print(f"Redundant models detected for grap_hash {graph_hash}:")
            for model_path in graph_paths:
                print(f"    {model_path}")
    assert (
        not find_redundant
    ), f"Redundant models detected under {args.graph_net_samples_path}."

    if args.model_path:
        assert os.path.isdir(
            args.model_path
        ), f"args.model_path {args.model_path} is not a directory!"
        current_model_graph_hash_pathes = set(
            graph_hash_path
            for model_path in get_recursively_model_pathes(args.model_path)
            for graph_hash_path in [f"{model_path}/graph_hash.txt"]
        )
        for current_model_graph_hash_path in current_model_graph_hash_pathes:
            graph_hash = open(current_model_graph_hash_path).read()
            assert (
                graph_hash not in graph_hash2graph_net_model_path
            ), f"Redundant models detected. old-model-path:{current_model_graph_hash_path}, new-model-path:{graph_hash2graph_net_model_path[graph_hash]}."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test compiler performance.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="Path to model file(s), each subdirectory containing graph_net.json will be regarded as a model",
    )
    parser.add_argument(
        "--graph-net-samples-path",
        type=str,
        required=True,
        help="Path to GraphNet samples",
    )
    args = parser.parse_args()
    print(args)
    main(args=args)
