from . import utils
import argparse
import importlib.util
import inspect
from graph_net.imp_util import load_module
import torch
import logging
from pathlib import Path
from typing import Type, Any
import sys
import json
import base64
from contextlib import contextmanager


def _load_class_from_file(file_path, class_name):
    module = load_module(file_path)
    return getattr(module, class_name)


def _convert_to_dict(config_str):
    if config_str is None:
        return {}
    config_str = base64.b64decode(config_str).decode("utf-8")
    config = json.loads(config_str)
    assert isinstance(config, dict), f"config should be a dict. {config_str=}"
    return config


def _get_checker(args):
    if args.checker_config is None:
        return lambda model_path: model_path
    checker_config = _convert_to_dict(args.checker_config).get(
        "post_extract_process_config"
    )
    checker_class = _load_class_from_file(
        checker_config["post_extract_process_path"],
        class_name=checker_config["post_extract_process_class_name"],
    )
    return checker_class(checker_config.get("checker_config", {}))


def main(args):
    checker = _get_checker(args)
    model_path = args.model_path
    print(f"{model_path=}")
    try:
        checker(model_path)
    except KeyboardInterrupt:
        sys.exit(-1)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load and run model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to folder e.g '../../samples/torch/resnet18'",
    )
    parser.add_argument(
        "--checker-config",
        type=str,
        required=False,
        default=None,
        help="checker configuration string",
    )
    args = parser.parse_args()
    main(args=args)
