from . import utils
import argparse
import base64
import importlib.util
import json
import os
import sys
from typing import Type

import torch
from jinja2 import Template


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


def get_flag_use_dummy_inputs(decorator_config):
    return "use_dummy_inputs" in decorator_config if decorator_config else False


def replay_tensor(info, use_dummy_inputs):
    if use_dummy_inputs:
        return utils.get_dummy_tensor(info)
    return utils.replay_tensor(info)


class AgentUnittestGenerator:
    """生成可独立运行的unittest脚本，用于子图前向可执行性验证。"""

    def __init__(self, config):
        defaults = {
            "model_path": None,
            "output_path": None,
            "force_device": "auto",  # auto / cpu / cuda
            "use_dummy_inputs": False,
        }
        merged = {**defaults, **(config or {})}
        if merged["model_path"] is None:
            raise ValueError("AgentUnittestGenerator requires 'model_path' in config")
        self.model_path = merged["model_path"]
        self.output_path = merged["output_path"] or self._default_output_path()
        self.force_device = merged["force_device"]
        self.use_dummy_inputs = merged["use_dummy_inputs"]

    def __call__(self, model):
        self._generate_unittest_file()
        return model

    def _default_output_path(self):
        base = os.path.basename(os.path.normpath(self.model_path))
        return os.path.join(self.model_path, f"{base}_test.py")

    def _choose_device(self):
        if self.force_device == "cpu":
            return "cpu"
        if self.force_device == "cuda":
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _generate_unittest_file(self):
        target_device = self._choose_device()
        template_str = """
import importlib.util
import os
import tempfile
import unittest

import torch
from graph_net.torch import utils


def _load_graph_module(model_path: str, target_device: str):
    source_path = os.path.join(model_path, "model.py")
    with open(source_path, "r", encoding="utf-8") as f:
        code = f.read()

    if target_device != "cuda":
        code = utils.modify_code_by_device(code, target_device)

    tmp_dir = tempfile.mkdtemp(prefix="agent_unittest_")
    tmp_file = os.path.join(tmp_dir, "model_tmp.py")
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(code)

    spec = importlib.util.spec_from_file_location("agent_graph_module", tmp_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraphModule


class AgentGraphTest(unittest.TestCase):
    def setUp(self):
        self.model_path = os.path.dirname(__file__)
        self.target_device = "{{ target_device }}"
        self.GraphModule = _load_graph_module(self.model_path, self.target_device)
        self.meta = utils.load_converted_from_text(self.model_path)
        self.use_dummy_inputs = {{ use_dummy_inputs }}

    def _with_device(self, info):
        cloned = {"info": dict(info["info"]), "data": info.get("data")}
        cloned["info"]["device"] = self.target_device
        return cloned

    def test_forward_runs(self):
        model = self.GraphModule()
        weight_info = self.meta["weight_info"]

        def _build_tensor(val):
            wrapped = self._with_device(val)
            return (
                utils.get_dummy_tensor(wrapped)
                if self.use_dummy_inputs
                else utils.replay_tensor(wrapped)
            )

        state_dict = {k: _build_tensor(v) for k, v in weight_info.items()}
        model.__graph_net_file_path__ = self.model_path
        output = model(**state_dict)
        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
"""

        rendered = Template(template_str).render(
            target_device=target_device, use_dummy_inputs=self.use_dummy_inputs
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
        decorator_config["decorator_config"].setdefault("use_dummy_inputs", False)

    model = _get_decorator(decorator_config)(model)

    inputs_params = utils.load_converted_from_text(f"{model_path}")
    params = inputs_params["weight_info"]
    use_dummy_inputs = get_flag_use_dummy_inputs(decorator_config)
    print(f"{use_dummy_inputs=}")
    state_dict = {k: replay_tensor(v, use_dummy_inputs) for k, v in params.items()}
    model.__graph_net_file_path__ = model_path
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
