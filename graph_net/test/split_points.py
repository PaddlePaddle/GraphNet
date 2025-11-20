import sys
import os
import argparse
import importlib.util
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Callable
from graph_net.torch import utils as graph_utils
from graph_net.torch.rp_expr.longest_rp_expr_parser import LongestRpExprParser
from graph_net.torch.rp_expr.rp_expr_parser import RpExprParser


class GraphExtractor:
    def __init__(self):
        self.extract_node = []

    def _extract_operators_from_graph(
        self, gm: nn.Module, example_inputs: List[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        operator_list = []
        named_modules = dict(gm.named_modules())

        for node in gm.graph.nodes:
            if node.op in ("call_method", "call_function", "call_module"):
                target_name = str(node.target)

                if node.op == "call_module":
                    module_instance = named_modules.get(node.target)
                    if module_instance is not None:
                        target_name = type(module_instance).__name__
                elif node.op == "call_function":
                    if isinstance(node.target, Callable):
                        target_name = node.target.__name__
                elif node.op == "call_method":
                    target_name = str(node.target)

                operator_info = {
                    "op_type": node.op,
                    "target": node.target,
                    "name": node.name,
                    "target_name": target_name,
                }
                operator_list.append(operator_info)

        return operator_list

    def extract_compiler(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
        operator = self._extract_operators_from_graph(gm, inputs)
        self.extract_node = operator
        return gm.forward


class ModelLoader:
    def load_class_from_file(self, model_path: str, device: str) -> Any:
        file_path = os.path.join(model_path, "model.py")
        file = Path(file_path).resolve()
        module_name = file.stem

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            model_code = f.read()

        model_code = graph_utils.modify_code_by_device(model_code, device)

        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        compiled_code = compile(model_code, filename=file, mode="exec")
        exec(compiled_code, module.__dict__)

        model_class = getattr(module, "GraphModule", None)
        if model_class is None:
            raise ImportError(f"Class 'GraphModule' not found in {file_path}")

        setattr(model_class, "__graph_net_file_path__", str(file_path))
        setattr(model_class, "__graph_net_device__", device)
        return model_class

    def get_input_dict(self, model_path: str, device: str) -> Dict[str, torch.Tensor]:
        inputs_params = graph_utils.load_converted_from_text(f"{model_path}")
        params = inputs_params["weight_info"]
        for tensor_meta in params.values():
            if hasattr(tensor_meta, "device"):
                tensor_meta.device = device
        input_dict = {
            k: graph_utils.replay_tensor(v).to(torch.device(device))
            for k, v in params.items()
        }
        return input_dict


def extract_ops_via_compile(model_path: str, device: str = "cpu") -> List[str]:
    loader = ModelLoader()
    print(f"Loading model from {model_path} on {device}...")
    try:
        model_class = loader.load_class_from_file(model_path, device)
        model = model_class().to(torch.device(device))
        model.eval()
        input_dict = loader.get_input_dict(model_path, device)
    except Exception as e:
        print(f"Error loading/preparing model {model_path}: {e}")
        return []

    extractor = GraphExtractor()
    compiled_model = torch.compile(model, backend=extractor.extract_compiler)

    with torch.no_grad():
        compiled_model(**input_dict)

    ops_info = extractor.extract_node
    if not ops_info:
        print(f"Warning: No operators extracted from {model_path}.")
        return []
    return [op["target_name"] for op in ops_info]


def calculate_token_lengths(rp_expr, num_primitives, symbol_map) -> Dict[int, int]:
    token2len = {}

    def get_len(tid):
        if tid in token2len:
            return token2len[tid]
        if tid < num_primitives:
            token2len[tid] = 1
            return 1
        if tid in symbol_map:
            sub_tokens = symbol_map[tid].tolist()
            length = sum(get_len(t) for t in sub_tokens)
            token2len[tid] = length
            return length
        token2len[tid] = 1
        return 1

    for sym_id in rp_expr.symbol_token_ids:
        get_len(sym_id)
    return token2len


def main():
    parser = argparse.ArgumentParser(
        description="Extract graph patterns and split points from multiple models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of paths to model directories (e.g. --models path/to/m1 path/to/m2)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    inputs = []
    valid_model_names = []

    for model_path in args.models:
        seq = extract_ops_via_compile(model_path, args.device)
        inputs.append(seq)
        valid_model_names.append(os.path.basename(model_path))

    rp_parser = RpExprParser(
        window_size=args.window, fold_policy="default", fold_times=0
    )
    rp_expr, token_id2primitive_id = rp_parser(inputs)

    rp_expr.try_unwrap_body_of_sole_symbol_token()
    rp_expr.try_recursive_inline_symbol_sole_used(token_id2primitive_id)

    num_primitives = len(token_id2primitive_id)
    symbol_map = dict(zip(rp_expr.symbol_token_ids, rp_expr.symbol_token_tensors))
    token2len = calculate_token_lengths(rp_expr, num_primitives, symbol_map)

    # print ops func
    def resolve_token_to_ops(tid) -> List[str]:
        if tid < num_primitives:
            return [token_id2primitive_id[tid]]
        if tid in symbol_map:
            sub_tokens = symbol_map[tid].tolist()
            ops = []
            for t in sub_tokens:
                ops.extend(resolve_token_to_ops(t))
            return ops
        return [f"Unknown({tid})"]

    for sym_id in sorted(symbol_map.keys()):
        length = token2len.get(sym_id, 0)
        ops_seq = resolve_token_to_ops(sym_id)
        ops_str = str(ops_seq)
        if len(ops_str) > 100:
            ops_str = ops_str[:100] + " ...]"

    for i, model_name in enumerate(valid_model_names):
        if i >= len(rp_expr.body_rp_expr):
            break

        target_body_tensor = rp_expr.body_rp_expr[i]
        seq_tokens = target_body_tensor.tolist()

        current_idx = 0
        split_points = set()
        total_len = sum(token2len.get(t, 1) for t in seq_tokens)

        full_model_ops = []
        for t in seq_tokens:
            full_model_ops.extend(resolve_token_to_ops(t))

        for token_id in seq_tokens:
            length = token2len.get(token_id, 1)
            is_pattern = token_id >= num_primitives

            if is_pattern:
                if current_idx > 0:
                    split_points.add(current_idx)
                end_idx = current_idx + length
                if end_idx < total_len:
                    split_points.add(end_idx)

            current_idx += length

        sorted_splits = sorted(list(set(split_points)))
        print("=" * 50)
        print(f"model_name: {model_name}")
        print(f"Split Sequence Indices: {sorted_splits}")
        print("Segments info:")
        last_split = 0
        for split in sorted_splits + [total_len]:
            segment_len = split - last_split
            if last_split < len(full_model_ops) and split <= len(full_model_ops):
                segment_ops = full_model_ops[last_split:split]
                if len(segment_ops) > 5:
                    ops_display = f"[{segment_ops[0]}, ..., {segment_ops[-1]}]"
                else:
                    ops_display = str(segment_ops)
                print(
                    f"  Range [{last_split:3d}, {split:3d}), Length: {segment_len:3d} | Ops: {ops_display}"
                )
            else:
                print(
                    f"  Range [{last_split:3d}, {split:3d}), Length: {segment_len:3d} | (Index Error Warning)"
                )

            last_split = split


if __name__ == "__main__":
    main()
