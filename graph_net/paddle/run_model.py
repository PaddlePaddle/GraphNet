import os
import json
import base64
import argparse
import numpy as np
import random

os.environ["FLAGS_logging_pir_py_code_dir"] = "/tmp/dump"

import paddle
from graph_net import imp_util
from graph_net.paddle import utils


def set_seed(random_seed):
    paddle.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def load_class_from_file(file_path: str, class_name: str):
    print(f"Load {class_name} from {file_path}")
    module = imp_util.load_module(file_path, "unnamed")
    model_class = getattr(module, class_name, None)
    setattr(model_class, "__graph_net_file_path__", os.path.normpath(file_path))
    return model_class


def get_input_dict(model_path):
    inputs_params = utils.load_converted_from_text(f"{model_path}")
    params = inputs_params["weight_info"]
    inputs = inputs_params["input_info"]

    input_dict = {}
    for name, meta in params.items():
        original_name = (
            meta["original_name"] if meta.get("original_name", None) else name
        )
        input_dict[name] = paddle.nn.parameter.Parameter(
            utils.replay_tensor(meta), name=original_name
        )
    for name, meta in inputs.items():
        input_dict[name] = utils.replay_tensor(meta)
    return input_dict


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
    initalize_seed = 123
    set_seed(random_seed=initalize_seed)

    model_path = args.model_path
    model_class = load_class_from_file(
        f"{model_path}/model.py", class_name="GraphModule"
    )
    assert model_class is not None
    model = model_class()
    print(f"{model_path=}")

    input_dict = get_input_dict(args.model_path)
    model = _get_decorator(args)(model)
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
