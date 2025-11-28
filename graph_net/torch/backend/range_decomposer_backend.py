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


class RangeDecomposerBackend:
    def __init__(self):
        self.graph_net_root = Path(graph_net.__file__).parent
        self.workspace_root = Path.cwd() / "naive_decompose_workspace"

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
