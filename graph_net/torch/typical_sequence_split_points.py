import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import tempfile
import graph_net.imp_util
from graph_net.torch import utils as graph_utils
from graph_net.torch.rp_expr.rp_expr_parser import RpExprParser


class TypicalSequenceExtractor:
    def __init__(self):
        self.extract_node = []

    def _extract_operators_from_graph(
        self, gm: nn.Module, example_inputs: List[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        operator_list = []
        named_modules = dict(gm.named_modules())

        for node in gm.graph.nodes:
            if node.op not in ("call_method", "call_function", "call_module"):
                continue

            if node.op == "call_module":
                target_name = type(named_modules[node.target]).__name__
            else:
                target_name = getattr(node.target, "__name__", str(node.target))

            operator_list.append(
                {
                    "op_type": node.op,
                    "target": node.target,
                    "name": node.name,
                    "target_name": target_name,
                }
            )

        return operator_list

    def extract_compiler(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]):
        operator = self._extract_operators_from_graph(gm, inputs)
        self.extract_node = operator
        return gm.forward


class TypicalSequenceModelLoader:
    def load_class_from_file(self, model_path: str, device: str) -> Any:
        file_path = os.path.join(model_path, "model.py")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            model_code = f.read()
        model_code = graph_utils.modify_code_by_device(model_code, device)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", encoding="utf-8"
        ) as temp_file:
            temp_file.write(model_code)
            module = graph_net.imp_util.load_module(temp_file.name)
        model_class = getattr(module, "GraphModule", None)

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


class SplitAnalyzer:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def _resolve_token_to_ops(
        self, tid, num_primitives, token_id2primitive_id, symbol_map
    ) -> List[str]:
        if tid < num_primitives:
            return [token_id2primitive_id[tid]]
        if tid in symbol_map:
            sub_tokens = symbol_map[tid].tolist()
            ops = []
            for t in sub_tokens:
                ops.extend(
                    self._resolve_token_to_ops(
                        t, num_primitives, token_id2primitive_id, symbol_map
                    )
                )
            return ops
        return [f"Unknown({tid})"]

    def _extract_ops_via_compile(
        self, model_path: str, device: str = "cpu"
    ) -> List[str]:
        loader = TypicalSequenceModelLoader()
        print(f"Loading model from {model_path} on {device}...")
        try:
            model_class = loader.load_class_from_file(model_path, device)
            model = model_class().to(torch.device(device))
            model.eval()
            input_dict = loader.get_input_dict(model_path, device)
        except Exception as e:
            print(f"Error loading/preparing model {model_path}: {e}")
            return []

        extractor = TypicalSequenceExtractor()
        compiled_model = torch.compile(model, backend=extractor.extract_compiler)
        compiled_model(**input_dict)
        ops_info = extractor.extract_node

        return [op["target_name"] for op in ops_info]

    def _calculate_token_lengths(
        self, rp_expr, num_primitives, symbol_map
    ) -> Dict[int, int]:
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

    def analyze(self, model_paths_file: str, device: str) -> Dict[str, Dict]:
        input_file = Path(model_paths_file)

        with open(input_file, "r") as f:
            model_paths = [
                Path(line.strip())
                for line in f
                if line.strip() and not line.startswith("#")
            ]

        inputs_seqs = []
        valid_models = []

        for p in model_paths:
            seq = self._extract_ops_via_compile(str(p), device)
            if seq:
                inputs_seqs.append(seq)
                valid_models.append((p.name, p))

        if not inputs_seqs:
            return {}

        rp_parser = RpExprParser(
            window_size=self.window_size, fold_policy="default", fold_times=0
        )
        rp_expr, token_id2primitive_id = rp_parser(inputs_seqs)
        rp_expr.try_unwrap_body_of_sole_symbol_token()
        rp_expr.try_recursive_inline_symbol_sole_used(token_id2primitive_id)

        num_primitives = len(token_id2primitive_id)
        symbol_map = dict(zip(rp_expr.symbol_token_ids, rp_expr.symbol_token_tensors))
        token2len = self._calculate_token_lengths(rp_expr, num_primitives, symbol_map)

        results = {}

        for i, (model_name, original_path) in enumerate(valid_models):
            if i >= len(rp_expr.body_rp_expr):
                break

            target_body_tensor = rp_expr.body_rp_expr[i]
            seq_tokens = target_body_tensor.tolist()

            full_model_ops = []
            for t in seq_tokens:
                full_model_ops.extend(
                    self._resolve_token_to_ops(
                        t, num_primitives, token_id2primitive_id, symbol_map
                    )
                )

            current_idx = 0
            split_points_set = set()
            total_len = sum(token2len.get(t, 1) for t in seq_tokens)

            for token_id in seq_tokens:
                length = token2len.get(token_id, 1)
                is_pattern = token_id >= num_primitives

                if is_pattern:
                    if current_idx > 0:
                        split_points_set.add(current_idx)
                    end_idx = current_idx + length
                    if end_idx < total_len:
                        split_points_set.add(end_idx)

                current_idx += length

            sorted_splits = sorted(list(split_points_set))

            self._print_analysis(
                model_name, str(original_path), sorted_splits, total_len, full_model_ops
            )

            results[model_name] = {
                "path": str(original_path),
                "split_points": sorted_splits,
                "total_length": total_len,
            }

        return results

    def _print_analysis(self, name, path, splits, total_len, full_ops):
        print("=" * 60)
        print(f"Model: {name}")
        print(f"Path:  {path}")
        print(f"Splits: {splits}")
        print("-" * 60)

        last_split = 0
        for split in splits + [total_len]:
            segment_len = split - last_split
            start_safe = min(last_split, len(full_ops))
            end_safe = min(split, len(full_ops))
            segment_ops = full_ops[start_safe:end_safe]

            ops_display = str(segment_ops)
            if len(segment_ops) > 5:
                ops_display = f"[{segment_ops[0]}, ..., {segment_ops[-1]}]"

            print(
                f"Range [{last_split:3d}, {split:3d}), Len: {segment_len:3d} | Ops: {ops_display}"
            )
            last_split = split
        print("\n")


def main(args):
    analyzer = SplitAnalyzer(window_size=args.window_size)
    results = analyzer.analyze(args.model_list, args.device)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze graph and calculate split points."
    )
    parser.add_argument(
        "--model-list",
        type=str,
        required=True,
        help="Path to a text file containing paths to models (one per line).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load models on (cpu, cuda).",
    )
    parser.add_argument(
        "--window-size", type=int, default=10, help="Window size for RP Parser."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="split_results.json",
        help="Path to save the analysis results in JSON format.",
    )
    args = parser.parse_args()
    main(args)
