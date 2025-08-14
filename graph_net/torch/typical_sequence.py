"""
Typical Sequence Extractor
Identify repeated subgraph patterns from extracted FX Graph and save them categorized.

python -m graph_net.torch.typical_sequence /path/to/model_path

输入：已经采集到的 模型的整图，在目录下体现为 6 个文件，核心是 model.py 和 weight_meta.py
输出：在 模型整图文件夹下的子目录 subgraph ，这个子目录中包含了一系列文件夹 typical_seq_<hash>_<idx>;  每个文件夹下都会包含类似的 6 个结构 
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
    MakeNestedIndexRangeFromLetsListTokenRpExp,
)


def compute_subgraph_hash(subgraph: List[Tuple[str, str]]) -> str:
    """
    Generate unique hash for subgraph based on op types and topology (ignore variable names).
    """
    return hashlib.sha256(...).hexdigest()[:8]


def find_forward_function(module_ast: ast.AST) -> ast.FunctionDef:
    """
    在 AST 中查找 GraphModule 的 forward 函数
    Find the 'forward' function in the AST of GraphModule class.
    """
    return None


def extract_assignments_from_forward(
    forward_func: ast.FunctionDef,
) -> List[Tuple[str, str]]:
    """
    Extract all assignment statements from forward function (simulating FX Graph node sequence).
    """
    return


def extract_used_parameters(subgraph: List[Tuple[str, str]]) -> List[str]:
    """
    Extract all parameter names used in subgraph (e.g., L_self_modules_...).
    """
    return


def extract_input_vars_from_subgraph(subgraph: List[Tuple[str, str]]) -> List[str]:
    """
    Extract input variables (those used but not defined in subgraph).
    """
    return


def generate_subgraph_model_py(subgraph: List[Tuple[str, str]]) -> str:
    """
    Generate model.py code for the subgraph.
    Returns a simplified GraphModule containing only this subgraph logic.
    """
    return


def generate_input_meta_py(inputs: List[str], weight_meta_path: str) -> str:
    """
    生成 input_meta.py (可复用原 weight_meta 中的结构）
    Generate input_meta.py (reuse structure from weight_meta).
    """
    return


def generate_tensor_constraints_py(inputs: List[str], weight_meta_path: str) -> str:
    """
    生成 input_tensor_constraints.py
    Generate input_tensor_constraints.py
    """
    return


def generate_weight_meta_py(used_params: List[str], weight_meta_path: str) -> str:
    """
    生成 weight_meta.py (仅包含子图用到的参数）
    Generate weight_meta.py (only used parameters).
    """
    return


def extract_typical_sequences(model_path: str, dynamic=True):
    """
    Extract typical subgraph sequences from a extracted model directory.

    Args:
        model_path (str): Model directory path, e.g., "/daiwenhao/graphnet_workspace/resnet18".
        dynamic (bool): Enable dynamic shape support in torch.compile.
    """

    model_py_path = os.path.join(model_path, "model.py")
    weight_meta_path = os.path.join(model_path, "weight_meta.py")
    if not os.path.exists(model_py_path):
        raise FileNotFoundError(f"Missing model.py: {model_py_path}")
    if not os.path.exists(weight_meta_path):
        raise FileNotFoundError(f"Missing weight_meta.py: {weight_meta_path}")

    # Parse model.py into AST and extract forward function of GraphModule
    with open(model_py_path, "r", encoding="utf-8") as f:
        source_code = f.read()
    module_ast = ast.parse(source_code)
    forward_func = find_forward_function(module_ast)

    # Extract assignment sequence (simulating FX Graph node flow) from AST
    assignments = extract_assignments_from_forward(forward_func)

    # Use RpExprParser to split into candidate subgraph sequences (to be implemented)
    rp_expr_parser = RpExprParser(window_size=64)
    lets_list_rp_expr, token_id2primitive_id = rp_expr_parser(assignments)
    print("\n".join(lets_list_rp_expr.DebugStrings(token_id2primitive_id)))
    candidate_subgraphs = MakeNestedIndexRangeFromLetsListTokenRpExpr(lets_list_rp_expr)

    # Generate structural hash for each subgraph (ignore var names, focus on op types & topology)
    subgraph_hashes = []
    for idx, subgraph in enumerate(candidate_subgraphs):
        graph_hash = compute_subgraph_hash(subgraph)
        subgraph_hashes.append((subgraph, graph_hash))

    # Group by hash value, same structure belongs to one class
    grouped_subgraphs: Dict[str, List[List[Tuple[str, str]]]] = {}
    for subgraph, h in subgraph_hashes:
        if h not in grouped_subgraphs:
            grouped_subgraphs[h] = []
        grouped_subgraphs[h].append(subgraph)

    # Create output directory: graphnet_workspace/resnet18/subgraph/
    subgraph_output_dir = os.path.join(model_path, "subgraph")
    os.makedirs(subgraph_output_dir, exist_ok=True)

    # 模仿整图目录结构: 创建 6 个文件
    # Create subgraph directories named typical_seq_<hash>_<idx>
    for graph_hash, instances in grouped_subgraphs.items():
        for instance_idx, subgraph in enumerate(instances):
            dir_name = f"typical_seq_{graph_hash}_{instance_idx}"
            instance_dir = os.path.join(subgraph_output_dir, dir_name)
            os.makedirs(instance_dir, exist_ok=True)

            # 1. Generate model.py for this subgraph instance
            subgraph_code = generate_subgraph_model_py(subgraph)
            with open(
                os.path.join(instance_dir, "model.py"), "w", encoding="utf-8"
            ) as f:
                f.write(subgraph_code)

            # 2. Generate graph_net.json (metadata, marked as subgraph)
            metadata = {
                "framework": "torch",
                "type": "subgraph",
                "parent_model": os.path.basename(model_path),
                "structure_hash": graph_hash,
                "instance_id": instance_idx,
                "dynamic": bool(dynamic),
            }
            with open(
                os.path.join(instance_dir, "graph_net.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(metadata, f, indent=4)

            # 3. Generate input_meta.py and input_tensor_constraints.py based on inputs
            inputs = extract_input_vars_from_subgraph(subgraph)
            input_meta_code = generate_input_meta_py(inputs, weight_meta_path)
            with open(
                os.path.join(instance_dir, "input_meta.py"), "w", encoding="utf-8"
            ) as f:
                f.write(input_meta_code)

            # 4. Generate input_tensor_constraints.py based on inputs
            constraints_code = generate_tensor_constraints_py(inputs, weight_meta_path)
            with open(
                os.path.join(instance_dir, "input_tensor_constraints.py"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(constraints_code)

            # 5. Generate weight_meta.py (parameters used in this subgraph)
            used_params = extract_used_parameters(subgraph)
            weight_meta_code = generate_weight_meta_py(used_params, weight_meta_path)
            with open(
                os.path.join(instance_dir, "weight_meta.py"), "w", encoding="utf-8"
            ) as f:
                f.write(weight_meta_code)

            # 对于整图抽取来说，是 validate 以后得到的 hash
            # 6. Optional: Generate graph hash file
            with open(os.path.join(instance_dir, "graph_hash.txt"), "w") as f:
                f.write(graph_hash + "\n")

    print(f"Subgraph extraction completed, results saved to...: {subgraph_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="resnet18")
    args = parser.parse_args()

    extract_typical_sequences(args.model_path)
