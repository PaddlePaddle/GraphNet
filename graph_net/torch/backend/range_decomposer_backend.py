import argparse
import base64
import importlib.util
import inspect
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn

import graph_net
from graph_net.torch import utils as graph_utils
from graph_net.torch.rp_expr.longest_rp_expr_parser import LongestRpExprParser
from graph_net.torch.rp_expr.rp_expr_parser import RpExprParser


def encode_config(config: Dict[str, Any]) -> str:
    json_str = json.dumps(config)
    return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")


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


class RangeDecomposerBackend:
    def __init__(self):
        self.window_size = 10
        self.graph_net_root = Path(graph_net.__file__).parent
        self.workspace_root = Path.cwd() / "naive_decompose_workspace"

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

    def _analyze_and_get_splits(self, args) -> Dict[str, Dict]:
        input_file = Path(args.model_path)
        if not input_file.exists():
            print(f"Error: Input file {input_file} does not exist.")
            return {}

        with open(input_file, "r") as f:
            model_paths = [
                Path(line.strip())
                for line in f
                if line.strip() and not line.startswith("#")
            ]

        if not model_paths:
            print("No valid model paths found.")
            return {}

        inputs_seqs = []
        valid_models = []

        for p in model_paths:
            seq = self._extract_ops_via_compile(p, args.device)
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
                model_name, original_path, sorted_splits, total_len, full_model_ops
            )

            results[model_name] = {
                "path": str(original_path),
                "split_points": sorted_splits,
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
                f"  Range [{last_split:3d}, {split:3d}), Len: {segment_len:3d} | Ops: {ops_display}"
            )
            last_split = split
        print("\n")

    def __call__(self, args):
        model_data_map = self._analyze_and_get_splits(args)

        for model_name, info in model_data_map.items():
            model_path = info["path"]
            split_points = info["split_points"]

            model_output_dir = self.workspace_root / f"{model_name}_decomposed"
            model_output_dir.mkdir(parents=True, exist_ok=True)

            config_dict = {
                "decorator_path": str(self.graph_net_root / "torch/extractor.py"),
                "decorator_config": {
                    "name": model_name,
                    "custom_extractor_path": str(
                        self.graph_net_root / "torch/naive_graph_decomposer.py"
                    ),
                    "custom_extractor_config": {
                        "output_dir": str(model_output_dir),
                        "split_positions": split_points,
                        "group_head_and_tail": True,
                        "filter_path": str(
                            self.graph_net_root / "torch/naive_subgraph_filter.py"
                        ),
                        "filter_config": {},
                    },
                },
            }

            encoded_config = encode_config(config_dict)

            cmd = [
                sys.executable,
                "-m",
                "graph_net.torch.run_model",
                "--model-path",
                model_path,
                "--decorator-config",
                encoded_config,
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"  [Success] Saved to {model_output_dir}")
            except subprocess.CalledProcessError as e:
                print(f"  [Error] Process failed: {e}")
            except Exception as e:
                print(f"  [Error] Unexpected: {e}")
