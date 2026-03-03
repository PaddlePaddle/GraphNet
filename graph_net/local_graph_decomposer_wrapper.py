import os
import argparse
import base64
import json
import subprocess
import sys
from typing import List
import tempfile
import shutil
import glob
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
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


def legacy_main(args):
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


@dataclass
class TmpSampleDesc:
    model_path_list: str
    model_path_prefix: str
    output_dir: str
    subgraph_ranges_json_root: str


def torch_main(args):
    with tmp_workspace(args) as tmp_sample_desc:
        return_code = subgraph_generate(tmp_sample_desc)
        _copy_outputs(tmp_sample_desc, args)
    sys.exit(return_code)


def _copy_outputs(tmp_sample_desc, args):
    prefix = (
        f"{tmp_sample_desc.output_dir}/{args.model_name}/_decomposed/{args.model_name}"
    )
    for model_path in glob.glob(f"{prefix}*"):
        last_underscore_pos = get_last_underscore_pos(model_path)
        last_third_underscore_pos = get_last_third_underscore_pos(model_path)
        assert (
            prefix == model_path[:last_third_underscore_pos]
        ), f"{prefix=} {model_path[:last_third_underscore_pos]=}"
        dst_model_path = f"{args.output_dir}/{args.model_name}/_decomposed/{args.model_name}{model_path[last_underscore_pos:]}"
        shutil.copytree(model_path, dst_model_path, dirs_exist_ok=True)
        print(
            f"Graph and tensors for '{args.model_name + model_path[last_underscore_pos:]}' extracted successfully to: {dst_model_path}"
        )


def get_last_underscore_pos(s):
    return s.rfind("_")


def get_last_third_underscore_pos(s):
    pos = s.rfind("_", 0)
    pos = s.rfind("_", 0, pos)
    pos = s.rfind("_", 0, pos)
    return pos


@contextmanager
def tmp_workspace(args):
    with tempfile.TemporaryDirectory() as temp_workspace:
        _copy_model_files(temp_workspace, args)
        _make_subgraph_ranges_json_file(temp_workspace, args)
        model_path_list = _make_model_path_list(temp_workspace, args)
        yield TmpSampleDesc(
            model_path_list=model_path_list,
            model_path_prefix=temp_workspace,
            output_dir=os.path.join(temp_workspace, "subgraphs"),
            subgraph_ranges_json_root=temp_workspace,
        )


def _copy_model_files(temp_workspace, args):
    dst_model_path = Path(temp_workspace) / args.model_name
    dst_model_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.model_path, dst_model_path, dirs_exist_ok=True)


def _make_subgraph_ranges_json_file(temp_workspace, args):
    json_obj = _make_subgraph_ranges_json(temp_workspace, args)
    dst_model_path = Path(temp_workspace) / args.model_name
    dst_model_path.mkdir(parents=True, exist_ok=True)
    with open(dst_model_path / "subgraph_ranges.json", "w") as f:
        json.dump(json_obj, f, indent=4)


def _make_subgraph_ranges_json(temp_workspace, args):
    split_positions = convert_b64_string_to_json(args.split_positions_json)
    assert isinstance(split_positions, list)
    subgraph_ranges = [
        (split_positions[i], split_positions[i + 1])
        for i in range(len(split_positions) - 1)
    ]
    return {
        "subgraph_ranges": subgraph_ranges,
    }


def _make_model_path_list(temp_workspace, args):
    dst_model_path = Path(temp_workspace) / args.model_name
    dst_model_path.mkdir(parents=True, exist_ok=True)
    dst_file_path = dst_model_path / "sample_list.txt"
    dst_file_path.write_text(f"{args.model_name}\n")
    return str(dst_file_path)


def subgraph_generate(tmp_sample_desc: TmpSampleDesc):
    sample_pass_config_config = {
        "model_path_prefix": tmp_sample_desc.model_path_prefix,
        "output_dir": tmp_sample_desc.output_dir,
        "subgraph_ranges_json_root": tmp_sample_desc.subgraph_ranges_json_root,
        "group_head_and_tail": False,
        "use_all_inputs": True,
        "chain_style": False,
        "resume": False,
    }
    sample_pass_config_b64 = convert_json_to_b64_string(sample_pass_config_config)

    cmd = [
        sys.executable,
        "-m",
        "graph_net.apply_sample_pass",
        "--model-path-list",
        tmp_sample_desc.model_path_list,
        "--sample-pass-file-path",
        f"{get_graphnet_root()}/graph_net/torch/sample_pass/subgraph_generator.py",
        "--sample-pass-class-name",
        "SubgraphGenerator",
        "--sample-pass-config",
        sample_pass_config_b64,
    ]

    print(" ".join(cmd))

    result = subprocess.run(cmd, text=True)
    return result.returncode


def main(args):
    entries = {
        "paddle": legacy_main,
        "torch": torch_main,
    }
    entries[args.framework](args)


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
    main(args)
