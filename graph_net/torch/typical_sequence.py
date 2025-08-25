"""
Typical Sequence Extractor
Identify repeated subgraph patterns from extracted FX Graph and save them categorized.
"""
import argparse
import os
import sys
import json
import hashlib
import ast
import copy
from pathlib import Path
from typing import List, Dict, Tuple, Set, Any
from graph_net.torch.rp_expr.rp_expr_parser import RpExprParser
from graph_net.torch.rp_expr.rp_expr_util import (
    MakeNestedIndexRangeFromLetsListTokenRpExpr,
)


def _get_leaf_model_pathes(src_model_path: str):
    # Traverse all submodule (features.0, classifier.6) in src_model_path
    return


def _get_fx_graph(leaf_model_path: str):
    # Load the GraphModule and extract its fx.Graph
    return


def _get_fx_node(fx_graph: str):
    # Traverse fx_graph.nodes to obtain all Nodes (splaceholder, call_function, output)
    return


def encode_node_to_stmt_token_id(node: str):
    # Node is encoded as token_id, representing a certain pattern
    return


def SequenceUnittestsGenerator(
    program_id: str, seq_stmts: List[str], dist_model_path: str
):
    # Generate unittests for each sequence
    return


def extract_typical_sequences(src_model_path: str, dist_model_path: str, dynamic=True):
    # Extract fx_graphs from src_model_path,  type: fx.graphmodule
    fx_graphs = [
        fx_graph
        for leaf_model_path in _get_leaf_model_pathes(src_model_path)
        for fx_graph in [_get_fx_graph(leaf_model_path)]
    ]

    # Convert each fx.Graph into a sequence of stmt token_ids
    stmt_token_ids = [
        [
            stmt_token_id
            for node in _get_fx_node(fx_graph)
            for stmt_token_id in [encode_node_to_stmt_token_id(node)]
        ]
        for fx_graph in fx_graphs
    ]

    # Extract typical subgraph patterns by RpExprParser
    parser = RpExprParser(window_size=64)
    lets_list_rp_expr, token_id2primitive_id = parser(stmt_token_ids)

    # Map the pattern back to the original FX Node range
    trees = MakeNestedIndexRangeFromLetsListTokenRpExpr(lets_list_rp_expr)

    # Filter by length
    ranges_list = [
        tree.FilterSubTreeRangeBySize(min_length=2, max_length=33) for tree in trees
    ]

    # get (program_id, seq_stmts) pair
    program_seq_stmts_list = (
        (program_id, seq_stmts)
        for ranges, pair in zip(ranges_list, stmt_token_ids)
        for start, end in ranges
        for program_id, origin_seq_stmts in [pair]
        for seq_stmts in [origin_seq_stmts[start:end]]
    )

    # Generate unittests for each sequence
    # Each folder: subgraph_<hash_id>_<count_id>.py
    SequenceUnittestsGenerator(program_id, seq_stmts, dist_model_path)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_model_path", type=str, default="samples/torchvision/alexnet"
    )
    parser.add_argument("--dist_model_path", type=str)
    args = parser.parse_args()

    # mkdir for dist model path; diff with src model path
    # Except:  src_model_path: GraphNet/samples/torchvision/alexnet
    #          dist_model_path: GraphNet/subgraphs/torchvision/alexnet
    if not args.dist_model_path:
        components = args.src_model_path.split(os.sep)
        try:
            samples_index = components.index("samples")
            components[samples_index] = "subgraphs"
        except ValueError:
            print(
                "Warning: 'samples' not found in src_model_path. Using default structure for dist_model_path."
            )
            components.insert(2, "subgraphs")
        args.dist_model_path = os.sep.join(components)
    os.makedirs(os.path.dirname(args.dist_model_path), exist_ok=True)

    # extract_typical_sequences(args.src_model_path, args.dist_model_path)

    print("Source model path:", args.src_model_path)
    print("Distribution model path:", args.dist_model_path)
