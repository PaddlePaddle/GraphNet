import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import torch
import torch.nn as nn
from graph_net.torch.rp_expr.rp_expr_parser import RpExprParser
from graph_net.torch.fx_graph_module_util import get_torch_module_and_inputs
from graph_net.torch.fx_graph_parse_util import parse_sole_graph_module_without_varify


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
            elif node.op == "call_method":
                target_name = f"Tensor.{node.target}"
            elif node.op == "call_function":
                target_name = getattr(node.target, "__name__", str(node.target))
            else:
                raise NotImplementedError()
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


class OpNamesExtractor:
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = self._make_config(**config)

    def _make_config(
        self, model_path_prefix: str, output_dir: str, resume: bool = False
    ):
        return {
            "model_path_prefix": model_path_prefix,
            "resume": resume,
            "output_dir": output_dir,
        }

    def __call__(self, rel_model_path: str):
        model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        output_path = self._get_output_path(rel_model_path)
        if self.config["resume"] and output_path.exists():
            return
        op_names = self._extract_ops(model_path)
        output_path.write_text("\n".join(op_names))
        print(f"Save op-names to {str(output_path)}")

    def _get_output_path(self, rel_model_path: str):
        output_path_dir = Path(self.config["output_dir"]) / rel_model_path
        output_path_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_path_dir / "op_names.txt"
        return output_path

    def _extract_ops(self, model_path: str) -> List[str]:
        extractor = TypicalSequenceExtractor()
        model, inputs = get_torch_module_and_inputs(model_path)
        compiled_model, _ = parse_sole_graph_module_without_varify(model, inputs)
        extractor.extract_compiler(compiled_model, inputs)
        ops_info = extractor.extract_node

        return [op["target_name"] for op in ops_info]


class SplitAnalyzer:
    def __init__(
        self, window_size: int = 10, fold_policy: str = "default", fold_times: int = 0
    ):
        self.window_size = window_size
        self.fold_policy = fold_policy
        self.fold_times = fold_times

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

    def _load_op_names_from_file(self, txt_path: Path) -> List[str]:
        if not txt_path.exists():
            print(f"File not found: {txt_path}")
            return []
        return txt_path.read_text().split("\n")

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

    def analyze(
        self, op_names_path_prefix: str, model_paths_file: str, device: str
    ) -> Dict[str, Dict]:
        input_file = Path(model_paths_file)

        with open(input_file, "r") as f:
            rel_model_paths = [
                Path(line.strip())
                for line in f
                if line.strip() and not line.startswith("#")
            ]

        inputs_seqs = []
        valid_models = []

        for rel_model_path in rel_model_paths:
            txt_path = Path(op_names_path_prefix) / rel_model_path / "op_names.txt"
            seq = self._load_op_names_from_file(txt_path)
            if seq:
                inputs_seqs.append(seq)
                valid_models.append((rel_model_path.name, rel_model_path))

        if not inputs_seqs:
            return {}
        rp_parser = RpExprParser(
            window_size=self.window_size,
            fold_policy=self.fold_policy,
            fold_times=self.fold_times,
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
            split_positions = set()
            total_len = sum(token2len.get(t, 1) for t in seq_tokens)

            for token_id in seq_tokens:
                length = token2len.get(token_id, 1)
                is_pattern = token_id >= num_primitives

                if is_pattern:
                    if current_idx > 0:
                        split_positions.add(current_idx)
                    end_idx = current_idx + length
                    if end_idx < total_len:
                        split_positions.add(end_idx)

                current_idx += length

            sorted_splits = sorted(list(split_positions))

            self._print_analysis(
                model_name, str(original_path), sorted_splits, total_len, full_model_ops
            )

            results[str(original_path)] = {
                "model_name": model_name,
                "split_positions": sorted_splits,
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


def _all_models_handled(args):
    output_json_path = Path(args.output_json)
    if not output_json_path.exists():
        return False
    with open(output_json_path) as f:
        output_json = json.load(f)
    rel_model_paths = [
        path for path in Path(args.model_list).read_text().split("\n") if len(path) > 0
    ]
    return all(path in output_json for path in rel_model_paths)


def main(args):
    if args.enable_resume and _all_models_handled(args):
        return
    analyzer = SplitAnalyzer(
        window_size=args.window_size,
        fold_policy=args.fold_policy,
        fold_times=args.fold_times,
    )
    results = analyzer.analyze(args.op_names_path_prefix, args.model_list, args.device)
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
        "--op-names-path-prefix",
        type=str,
        default="./",
        help="Prefix to add to each op_names.txt file path in the list.",
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
        "--fold-policy",
        type=str,
        default="default",
        help="Policy for split analysis, one of 'default' or 'longest'",
    )
    parser.add_argument(
        "--fold-times",
        type=int,
        default=0,
        help="How many times to fold tokens. If 0, then no folding is done.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="split_results.json",
        help="Path to save the analysis results in JSON format.",
    )
    parser.add_argument(
        "--enable-resume",
        action="store_true",
        default=False,
        help="Resume process",
    )
    args = parser.parse_args()
    main(args)
