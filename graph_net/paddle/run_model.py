import argparse
import base64
import importlib
import importlib.util
import json
import os
import sys

os.environ["FLAGS_logging_pir_py_code_dir"] = "/tmp/dump"

import paddle
from graph_net import imp_util
from graph_net.paddle import utils


BUILTIN_DECORATORS = {
    "AgentUnittestGenerator": "graph_net.paddle.sample_passes.agent_unittest_generator",
}


def load_class_from_file(file_path: str, class_name: str):
    print(f"Load {class_name} from {file_path}")
    module = imp_util.load_module(file_path, "unnamed")
    model_class = getattr(module, class_name, None)
    return model_class


def _load_builtin_decorator(class_name: str):
    module_path = BUILTIN_DECORATORS.get(class_name)
    if not module_path:
        return None
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return None
    return getattr(module, class_name, None)


def get_input_dict(model_path):
    inputs_params = utils.load_converted_from_text(f"{model_path}")
    params = inputs_params["weight_info"]
    inputs = inputs_params["input_info"]

    state_dict = {}
    for k, v in params.items():
        state_dict[k] = paddle.nn.parameter.Parameter(utils.replay_tensor(v), name=k)
    for k, v in inputs.items():
        state_dict[k] = utils.replay_tensor(v)
    return state_dict


def _convert_to_dict(config_str):
    if config_str is None:
        return {}
    config_str = base64.b64decode(config_str).decode("utf-8")
    config = json.loads(config_str)
    assert isinstance(config, dict), f"config should be a dict. {config_str=}"
    return config


def _get_decorator(arg):
    """Accept argparse.Namespace or already-parsed dict configs."""
    if arg is None:
        return lambda model: model

    decorator_config = (
        _convert_to_dict(arg.decorator_config)
        if hasattr(arg, "decorator_config")
        else arg
    )
    if not decorator_config:
        return lambda model: model

    class_name = decorator_config.get("decorator_class_name", "RunModelDecorator")
    decorator_kwargs = decorator_config.get("decorator_config", {})

    if "decorator_path" in decorator_config:
        decorator_class = load_class_from_file(
            decorator_config["decorator_path"], class_name=class_name
        )
        return decorator_class(decorator_kwargs)

    builtin_decorator = _load_builtin_decorator(class_name)
    if builtin_decorator:
        return builtin_decorator(decorator_kwargs)

    if hasattr(sys.modules[__name__], class_name):
        decorator_class = getattr(sys.modules[__name__], class_name)
        return decorator_class(decorator_kwargs)

    return lambda model: model


def main(args):
    model_path = args.model_path
    model_class = load_class_from_file(
        f"{model_path}/model.py", class_name="GraphModule"
    )
    assert model_class is not None
    model = model_class()
    print(f"{model_path=}")
    decorator_config = _convert_to_dict(args.decorator_config)
    if decorator_config:
        decorator_config.setdefault("decorator_config", {})
        decorator_config["decorator_config"].setdefault("model_path", model_path)
        decorator_config["decorator_config"].setdefault("output_dir", None)
        decorator_config["decorator_config"].setdefault("use_numpy", True)

    model = _get_decorator(decorator_config)(model)
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
