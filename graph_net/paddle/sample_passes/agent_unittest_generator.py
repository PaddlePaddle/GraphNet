from pathlib import Path
from typing import Any, Dict

import paddle
from jinja2 import Template

from graph_net.sample_pass.sample_pass import SamplePass


PADDLE_UNITTEST_TEMPLATE = r"""
import importlib.util
import os
import unittest
from typing import Any, Dict

import numpy as np
import paddle


def _get_classes(file_path: str):
    spec = importlib.util.spec_from_file_location("agent_meta", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [
        (name, cls)
        for name, cls in vars(module).items()
        if isinstance(cls, type)
    ]


def _convert_meta_classes_to_wrappers(file_path: str):
    current_device = paddle.device.get_device()
    for _, cls in _get_classes(file_path):
        attrs = {
            k: v for k, v in vars(cls).items() if not k.startswith("__") and not callable(v)
        }
        dtype_attr = attrs.get("dtype", "float32")
        dtype = getattr(paddle, str(dtype_attr).split(".")[-1])
        shape = [1 if dim is None else dim for dim in attrs.get("shape", [])]
        info = {
            "shape": shape,
            "dtype": dtype,
            "device": attrs.get("device", current_device),
            "mean": attrs.get("mean"),
            "std": attrs.get("std"),
            "min_val": attrs.get("min_val", 0),
            "max_val": attrs.get("max_val", 2),
        }
        data = attrs.get("data")
        if data is not None and not isinstance(data, paddle.Tensor):
            data = paddle.to_tensor(data, dtype=dtype).reshape(info["shape"])
        yield {"info": info, "data": data, "name": attrs.get("name")}


def _convert_meta_to_tensors(model_path: str):
    weight_meta = os.path.join(model_path, "weight_meta.py")
    input_meta = os.path.join(model_path, "input_meta.py")
    weight_info = {
        item["name"]: item for item in _convert_meta_classes_to_wrappers(weight_meta)
    }
    input_info = {
        item["name"]: item for item in _convert_meta_classes_to_wrappers(input_meta)
    }
    return {"weight_info": weight_info, "input_info": input_info}


def _init_integer_tensor(dtype, shape, min_val, max_val, use_numpy: bool):
    if use_numpy:
        array = np.random.randint(low=min_val, high=max_val + 1, size=shape, dtype=dtype)
        return paddle.to_tensor(array)
    return paddle.randint(low=min_val, high=max_val + 1, shape=shape, dtype=dtype)


def _init_float_tensor(shape, mean, std, min_val, max_val, use_numpy: bool):
    if use_numpy:
        if mean is not None and std is not None:
            array = np.random.normal(0, 1, shape) * std * 0.2 + mean
            array = np.clip(array, min_val, max_val)
        else:
            array = np.random.uniform(low=min_val, high=max_val, size=shape)
        return paddle.to_tensor(array)
    if mean is not None and std is not None:
        tensor = paddle.randn(shape, dtype="float32") * std * 0.2 + mean
        tensor = paddle.clip(tensor, min=min_val, max=max_val)
        return tensor
    return paddle.uniform(shape=shape, dtype="float32", min=min_val, max=max_val)


def _replay_tensor(info: Dict[str, Any], use_numpy: bool):
    device = info["info"].get("device", paddle.device.get_device())
    dtype = info["info"].get("dtype", paddle.float32)
    shape = [1 if dim is None else dim for dim in info["info"].get("shape", [])]
    mean = info["info"].get("mean")
    std = info["info"].get("std")
    min_val = info["info"].get("min_val", 0)
    max_val = info["info"].get("max_val", 2)
    if info.get("data") is not None:
        return paddle.reshape(info["data"], shape).to(dtype).to(device)
    if dtype in [paddle.int32, paddle.int64, paddle.bool]:
        init_dtype = "int32" if dtype == paddle.bool else "int64"
        if dtype == paddle.bool:
            min_val, max_val = 0, 1
        return _init_integer_tensor(init_dtype, shape, min_val, max_val, use_numpy).to(dtype).to(device)
    tensor = _init_float_tensor(shape, mean, std, min_val, max_val, use_numpy)
    return tensor.to(dtype).to(device)


def _get_dummy_tensor(info: Dict[str, Any]):
    device = info["info"].get("device", paddle.device.get_device())
    dtype = info["info"].get("dtype", paddle.float32)
    shape = [1 if dim is None else dim for dim in info["info"].get("shape", [])]
    if info.get("data") is not None:
        return paddle.reshape(info["data"], shape).to(dtype).to(device)
    return paddle.empty(shape=shape, dtype=dtype, device=device)


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
        self.use_numpy = {{ use_numpy_flag }}
        paddle.set_device(self.target_device)
        self.GraphModule = _load_graph_module(self.model_path)
        self.meta = _convert_meta_to_tensors(self.model_path)

    def _with_device(self, info: Dict[str, Any]):
        cloned = {"info": dict(info["info"]), "data": info.get("data")}
        cloned["info"]["device"] = self.target_device
        return cloned

    def test_forward_runs(self):
        model = self.GraphModule()
        inputs = {k: _replay_tensor(self._with_device(v), self.use_numpy) for k, v in self.meta["input_info"].items()}
        params = {k: _replay_tensor(self._with_device(v), self.use_numpy) for k, v in self.meta["weight_info"].items()}
        model.__graph_net_file_path__ = self.model_path
        output = model(**params, **inputs)
        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
"""


class AgentUnittestGenerator:
    """Generate standalone unittest scripts for Paddle samples."""

    def __init__(self, config: Dict[str, Any]):
        defaults = {
            "model_path": None,
            "output_path": None,
            "output_dir": None,
            "force_device": "auto",  # auto / cpu / gpu
            "use_numpy": True,
        }
        merged = {**defaults, **(config or {})}
        if not merged["model_path"]:
            raise ValueError("AgentUnittestGenerator requires 'model_path' in config")

        self.model_path = Path(merged["model_path"]).resolve()
        self.output_path = (
            Path(merged["output_path"]) if merged.get("output_path") else None
        )
        self.output_dir = (
            Path(merged["output_dir"]) if merged.get("output_dir") else None
        )
        self.force_device = merged["force_device"]
        self.use_numpy = merged["use_numpy"]

    def __call__(self, model):
        self.generate()
        return model

    def generate(self):
        output_path = self._resolve_output_path()
        target_device = self._choose_device()
        rendered = Template(PADDLE_UNITTEST_TEMPLATE).render(
            target_device=target_device, use_numpy_flag=self.use_numpy
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"[Agent] unittest generated: {output_path} (device={target_device})")

    def _resolve_output_path(self) -> Path:
        if self.output_path:
            return self.output_path
        target_dir = self.output_dir or self.model_path
        return Path(target_dir) / f"{self.model_path.name}_test.py"

    def _choose_device(self) -> str:
        if self.force_device == "cpu":
            return "cpu"
        if self.force_device == "gpu":
            return "gpu"
        return "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"


class AgentUnittestGeneratorPass(SamplePass):
    """SamplePass wrapper to generate Paddle unittests via model_path_handler."""

    def __init__(self, config=None):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str = None,
        force_device: str = "auto",
        use_numpy: bool = True,
    ):
        pass

    def __call__(self, rel_model_path: str):
        model_path_prefix = Path(self.config["model_path_prefix"])
        target_root = Path(self.config.get("output_dir") or model_path_prefix)
        model_path = model_path_prefix / rel_model_path
        generator = AgentUnittestGenerator(
            {
                "model_path": str(model_path),
                "output_dir": str(target_root / rel_model_path),
                "force_device": self.config["force_device"],
                "use_numpy": self.config["use_numpy"],
            }
        )
        generator.generate()
