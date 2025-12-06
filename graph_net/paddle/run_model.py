import argparse
import base64
import importlib.util
import json
import os
import sys

os.environ["FLAGS_logging_pir_py_code_dir"] = "/tmp/dump"

import paddle
from graph_net import imp_util
from graph_net.paddle import utils
from jinja2 import Template


def load_class_from_file(file_path: str, class_name: str):
    print(f"Load {class_name} from {file_path}")
    module = imp_util.load_module(file_path, "unnamed")
    model_class = getattr(module, class_name, None)
    return model_class


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
    """兼容旧接口：既接受 argparse.Namespace，也接受已解析的 dict。"""
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

    if hasattr(sys.modules[__name__], class_name):
        decorator_class = getattr(sys.modules[__name__], class_name)
        return decorator_class(decorator_kwargs)

    return lambda model: model


class AgentUnittestGenerator:
    """生成 Paddle 子图的独立 unittest 脚本，验证前向可运行。"""

    def __init__(self, config):
        defaults = {
            "model_path": None,
            "output_path": None,
            "force_device": "auto",  # auto / cpu / gpu
            "use_numpy": True,
        }
        merged = {**defaults, **(config or {})}
        if merged["model_path"] is None:
            raise ValueError("AgentUnittestGenerator requires 'model_path' in config")
        self.model_path = merged["model_path"]
        self.output_path = merged["output_path"] or self._default_output_path()
        self.force_device = merged["force_device"]
        self.use_numpy = merged["use_numpy"]

    def __call__(self, model):
        self._generate_unittest_file()
        return model

    def _default_output_path(self):
        base = os.path.basename(os.path.normpath(self.model_path))
        return os.path.join(self.model_path, f"{base}_test.py")

    def _choose_device(self):
        if self.force_device == "cpu":
            return "cpu"
        if self.force_device == "gpu":
            return "gpu"
        return "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"

    def _generate_unittest_file(self):
        target_device = self._choose_device()
        template_str = """
import importlib.util
import os
import unittest

import paddle
from graph_net.paddle import utils


def _load_graph_module(model_path: str):
    source_path = os.path.join(model_path, "model.py")
    spec = importlib.util.spec_from_file_location("agent_graph_module", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraphModule


class AgentGraphTest(unittest.TestCase):
    def setUp(self):
        self.model_path = os.path.dirname(__file__)
        self.target_device = "{{ target_device }}"
        paddle.set_device(self.target_device)
        self.GraphModule = _load_graph_module(self.model_path)
        self.meta = utils.load_converted_from_text(self.model_path)
        self.use_numpy = {{ use_numpy_flag }}

    def _with_device(self, info):
        cloned = {"info": dict(info["info"]), "data": info.get("data")}
        cloned["info"]["device"] = self.target_device
        return cloned

    def _build_tensor(self, meta):
        return utils.replay_tensor(self._with_device(meta), use_numpy=self.use_numpy)

    def test_forward_runs(self):
        model = self.GraphModule()
        inputs = {k: self._build_tensor(v) for k, v in self.meta["input_info"].items()}
        params = {k: self._build_tensor(v) for k, v in self.meta["weight_info"].items()}
        model.__graph_net_file_path__ = self.model_path
        output = model(**params, **inputs)
        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
"""

        rendered = Template(template_str).render(
            target_device=target_device, use_numpy_flag=self.use_numpy
        )

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(rendered)
        print(f"[Agent] unittest 已生成: {self.output_path} (device={target_device})")


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
