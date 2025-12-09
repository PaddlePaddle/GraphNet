import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import graph_net


def encode_config(config: Dict[str, Any]) -> str:
    json_str = json.dumps(config)
    return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data_dict = json.load(file)
    return data_dict


class RangeDecomposerBackend:
    def __init__(self):
        self.graph_net_root = Path(graph_net.__file__).parent

    def __call__(self, model: torch.nn.Module) -> torch.nn.Module:
        config = self.config
        workspace_path = Path(config["workspace_path"])
        chain_style = config["chain_style"]

        model_file_path = Path(model.__class__.__graph_net_file_path__)
        model_name = model_file_path.parent.name

        model_info = load_json(config["split_results_path"])[model_name]
        model_path = model_info["path"]
        split_points = model_info["split_points"]

        model_output_dir = workspace_path / f"{model_name}_decomposed"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "decorator_path": str(self.graph_net_root / "torch/extractor.py"),
            "decorator_config": {
                "name": model_name,
                "custom_extractor_path": str(
                    self.graph_net_root / "torch/graph_decomposer.py"
                ),
                "custom_extractor_config": {
                    "output_dir": str(model_output_dir),
                    "split_positions": split_points,
                    "group_head_and_tail": True,
                    "filter_path": str(
                        self.graph_net_root / "torch/naive_subgraph_filter.py"
                    ),
                    "filter_config": {},
                    "chain_style": chain_style,
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
            print(f"[Success] Saved to {model_output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"[Error] Process failed: {e}")
        except Exception as e:
            print(f"[Error] Unexpected: {e}")
        return model

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
