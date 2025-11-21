import sys
import json
import base64
import argparse
import importlib.util
from typing import Type

import paddle
from graph_net.paddle import utils


def load_class_from_file(file_path: str, class_name: str):
    print(f"Load {class_name} from {file_path}")
    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)
    model_class = getattr(unnamed, class_name, None)
    return model_class


def get_input_dict(model_path):
    inputs_params = utils.load_converted_from_text(f"{model_path}")
    params = inputs_params["weight_info"]
    inputs = inputs_params["input_info"]

    params.update(inputs)
    state_dict = {k: utils.replay_tensor(v) for k, v in params.items()}
    return state_dict


def _convert_to_dict(config_str):
    if config_str is None:
        return {}
    config_str = base64.b64decode(config_str).decode("utf-8")
    config = json.loads(config_str)
    assert isinstance(config, dict), f"config should be a dict. {config_str=}"
    return config


def _get_decorator(args):
    if args.decorator_config is None:
        return lambda model: model
    decorator_config = _convert_to_dict(args.decorator_config)
    if "decorator_path" not in decorator_config:
        return lambda model: model
    decorator_class = load_class_from_file(
        decorator_config["decorator_path"], class_name="RunModelDecorator"
    )
    return decorator_class(decorator_config.get("decorator_config", {}))


def main(args):
    model_path = args.model_path
    model_class = load_class_from_file(
        f"{model_path}/model.py", class_name="GraphModule"
    )
    assert model_class is not None
    model = model_class()
    print(f"{model_path=}")

    model = _get_decorator(args)(model)
    input_dict = get_input_dict(args.model_path)

    model(**input_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load and run model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to folder e.g '../../paddle_samples/PaddleX/ResNet18'",
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
