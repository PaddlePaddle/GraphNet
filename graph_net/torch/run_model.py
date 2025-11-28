from . import utils
import argparse
import importlib.util
import inspect
import torch
import logging
from pathlib import Path
from typing import Type, Any
import sys
import json
import base64
from contextlib import contextmanager


def load_class_from_file(file_path: str, class_name: str) -> Type[torch.nn.Module]:
    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)
    model_class = getattr(unnamed, class_name, None)
    setattr(model_class, "__graph_net_file_path__", file_path)
    return model_class


def _convert_to_dict(config_str):
    if config_str is None:
        return {}
    config_str = base64.b64decode(config_str).decode("utf-8")
    config = json.loads(config_str)
    assert isinstance(config, dict), f"config should be a dict. {config_str=}"
    return config


def _get_decorator(decorator_config):
    if "decorator_path" not in decorator_config:
        return lambda model: model
    class_name = decorator_config.get("decorator_class_name", "RunModelDecorator")
    decorator_class = load_class_from_file(
        decorator_config["decorator_path"],
        class_name=class_name,
    )
    return decorator_class(decorator_config.get("decorator_config", {}))


def get_flag_use_dummy_inputs(decorator_config):
    return "use_dummy_inputs" in decorator_config


def replay_tensor(info, use_dummy_inputs):
    if use_dummy_inputs:
        return utils.get_dummy_tensor(info)
    else:
        return utils.replay_tensor(info)


def main(args):
    model_path = args.model_path
    model_class = load_class_from_file(
        f"{model_path}/model.py", class_name="GraphModule"
    )
    assert model_class is not None
    model = model_class()
    print(f"{model_path=}")

    decorator_config = _convert_to_dict(args.decorator_config)
    if "decorator_path" in args.decorator_config:
        model = _get_decorator(decorator_config)(model)

    inputs_params = utils.load_converted_from_text(f"{model_path}")
    params = inputs_params["weight_info"]
    use_dummy_inputs = get_flag_use_dummy_inputs(decorator_config)
    print(f"{use_dummy_inputs=}")
    state_dict = {k: replay_tensor(v, use_dummy_inputs) for k, v in params.items()}

    model(**state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load and run model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to folder e.g '../../samples/torch/resnet18'",
    )
    parser.add_argument(
        "--decorator-config",
        type=str,
        required=False,
        default=None,
        help="decorator configuration string",
    )
    args = parser.parse_args()
    main(args=args)
