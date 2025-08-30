from . import utils
import argparse
import importlib.util
import inspect
from pathlib import Path
from typing import Type, Any
import sys
import hashlib
from contextlib import contextmanager
from collections import ChainMap
import numpy as np
import graph_net
import os
import re
import paddle


def load_class_from_file(file_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)
    model_class = getattr(unnamed, class_name, None)
    return model_class


def _get_sha_hash(content):
    m = hashlib.sha256()
    m.update(content.encode())
    return m.hexdigest()


def _save_to_model_path(dump_dir, hash_text):
    file_path = f"{dump_dir}/graph_hash.txt"
    with open(file_path, "w") as f:
        f.write(hash_text)


def extract_from_forward_regex(text, case_sensitive=True):
    pattern = r"forward.*"
    flags = 0 if case_sensitive else re.IGNORECASE

    match = re.search(pattern, text, flags)
    if match:
        return match.group(0)
    else:
        raise ValueError("Erroneous case occurs.")


def main(args):
    model_path = args.model_path
    with open(f"{model_path}/model.py", "r") as fp:
        model_str = fp.read()
        model_str = extract_from_forward_regex(model_str)
        _save_to_model_path(model_path, _get_sha_hash(model_str))

    model_path = args.model_path
    model_class = load_class_from_file(
        f"{model_path}/model.py", class_name="GraphModule"
    )
    assert model_class is not None
    model = model_class()
    inputs_params = utils.load_converted_from_text(f"{model_path}")
    params = inputs_params["weight_info"]
    inputs = inputs_params["input_info"]

    params.update(inputs)
    state_dict = {k: utils.replay_tensor(v) for k, v in params.items()}

    y = model(**state_dict)

    # print(np.argmin(y), np.argmax(y))
    if isinstance(y, paddle.Tensor):
        print(y.shape)
    elif isinstance(y, list) or isinstance(y, tuple):
        print(y[0].shape if isinstance(y[0], paddle.Tensor) else y[0])
    else:
        raise ValueError("Illegal return value.")

    if not args.no_check_redundancy:
        print("Check redundancy ...")
        graph_net_samples_path = (
            graph_net.paddle.samples_util.get_default_samples_directory()
            if args.graph_net_samples_path is None
            else args.graph_net_samples_path
        )
        cmd = f"{sys.executable} -m graph_net.paddle.check_redundant_incrementally --model-path {args.model_path} --graph-net-samples-path {graph_net_samples_path}"
        cmd_ret = os.system(cmd)
        rm_cmd = f"{sys.executable} -m graph_net.paddle.remove_redundant_incrementally --model-path {args.model_path} --graph-net-samples-path {graph_net_samples_path}"
        assert (
            cmd_ret == 0
        ), f"\nPlease use the following command to remove redundant model directories:\n\n{rm_cmd}\n"

    print(f"Validation success, {model_path=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load and run model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to folder e.g '../test_dataset'",
    )

    parser.add_argument(
        "--no-check-redundancy",
        action="store_true",
        help="whether check model graph redundancy",
    )

    parser.add_argument(
        "--dump-graph-hash-key",
        action="store_true",
        default=False,
        help="Dump graph hash key",
    )
    parser.add_argument(
        "--graph-net-samples-path",
        type=str,
        required=False,
        default=None,
        help="Path to GraphNet samples folder. e.g '../../samples'",
    )
    args = parser.parse_args()
    main(args=args)
