import argparse
import base64
import json
import subprocess
import sys
from typing import List

from graph_net.graph_net_root import get_graphnet_root


def convert_b64_string_to_json(b64str):
    return json.loads(base64.b64decode(b64str).decode("utf-8"))


def convert_json_to_b64_string(config) -> str:
    return base64.b64encode(json.dumps(config).encode()).decode()


def build_decorator_config(
    framework: str,
    model_name: str,
    output_dir: str,
    split_positions: List[int],
) -> dict:
    graphnet_root = get_graphnet_root()
    decorator_config = {
        "decorator_path": f"{graphnet_root}/graph_net/{framework}/extractor.py",
        "decorator_config": {
            "name": model_name,
            "custom_extractor_path": f"{graphnet_root}/graph_net/{framework}/graph_decomposer.py",
            "custom_extractor_config": {
                "output_dir": output_dir,
                "split_positions": split_positions,
                "group_head_and_tail": False,
                "use_all_inputs": True,
                "chain_style": False,
            },
        },
    }

    if framework == "paddle":
        post_process_configs = {
            "post_extract_process_path": f"{graphnet_root}/graph_net/{framework}/graph_meta_restorer.py",
            "post_extract_process_class_name": "GraphMetaRestorer",
            "post_extract_process_config": {
                "update_inplace": True,
                "input_meta_allow_partial_update": False,
            },
        }
        decorator_config["decorator_config"]["custom_extractor_config"].update(
            post_process_configs
        )

    return decorator_config


def main():
    split_positions = convert_b64_string_to_json(args.split_positions_json)
    if not isinstance(split_positions, list) or not all(
        isinstance(x, int) for x in split_positions
    ):
        raise ValueError(f"Invalid split positions: {split_positions}")

    decorator_config = build_decorator_config(
        framework=args.framework,
        model_name=args.model_name,
        output_dir=args.output_dir,
        split_positions=split_positions,
    )
    decorator_config_b64 = convert_json_to_b64_string(decorator_config)

    cmd = [
        sys.executable,
        "-m",
        f"graph_net.{args.framework}.run_model",
        "--model-path",
        args.model_path,
        "--decorator-config",
        decorator_config_b64,
    ]

    result = subprocess.run(cmd, text=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework", type=str, choices=["paddle", "torch"], required=True
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-positions-json", type=str, required=True)

    args = parser.parse_args()
    main()
