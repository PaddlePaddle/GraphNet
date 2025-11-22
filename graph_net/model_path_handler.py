import argparse
import importlib.util
from graph_net.imp_util import load_module
import inspect
import logging
from pathlib import Path
from typing import Type, Any
import sys
import json
import base64
from contextlib import contextmanager


def _convert_to_dict(config_str):
    if config_str is None:
        return {}
    config_str = base64.b64decode(config_str).decode("utf-8")
    config = json.loads(config_str)
    assert isinstance(config, dict), f"config should be a dict. {config_str=}"
    return config


def _load_class_from_file(file_path, class_name):
    module = load_module(file_path)
    return getattr(module, class_name)


def _get_handler(args):
    if args.handler_config is None:
        return lambda model_path: model_path
    handler_config = _convert_to_dict(args.handler_config)
    handler_class = _load_class_from_file(
        handler_config["handler_path"], class_name=handler_config["handler_class_name"]
    )
    return handler_class(handler_config.get("handler_config", {}))


def main(args):
    model_path = args.model_path
    print(f"{model_path=}")

    _get_handler(args)(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model path handler entry")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to folder e.g '../../samples/torch/resnet18'",
    )
    parser.add_argument(
        "--handler-config",
        type=str,
        required=False,
        default=None,
        help="handler configuration string",
    )
    args = parser.parse_args()
    main(args=args)
