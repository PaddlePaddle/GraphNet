from pathlib import Path
from typing import Any, Dict

import torch
from jinja2 import Template

from graph_net.sample_pass.sample_pass import SamplePass


TORCH_UNITTEST_TEMPLATE = r"""
import ast
import importlib.util
import os
import tempfile
import unittest
from typing import Any, Dict

import torch


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
    for _, cls in _get_classes(file_path):
        attrs = {
            k: v for k, v in vars(cls).items() if not k.startswith("__") and not callable(v)
        }
        dtype_attr = attrs.get("dtype", "torch.float32")
        dtype = getattr(torch, str(dtype_attr).split(".")[-1])
        shape = [1 if dim is None else dim for dim in attrs.get("shape", [])]
        info = {
            "shape": shape,
            "dtype": dtype,
            "device": attrs.get("device", "cpu"),
            "mean": attrs.get("mean", 0.0),
            "std": attrs.get("std", 1.0),
            "min_val": attrs.get("min_val"),
            "max_val": attrs.get("max_val"),
        }
        data = attrs.get("data")
        if data is not None and not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=dtype).reshape(info["shape"])
        yield {"info": info, "data": data, "name": attrs.get("name")}


def _convert_meta_to_tensors(model_path: str):
    weight_meta = os.path.join(model_path, "weight_meta.py")
    input_meta = os.path.join(model_path, "input_meta.py")
    weight_info = {
        item["name"]: item for item in _convert_meta_classes_to_wrappers(weight_meta)
    }
    input_info = list(_convert_meta_classes_to_wrappers(input_meta))
    return {"weight_info": weight_info, "input_info": input_info}


def _replay_tensor(info: Dict[str, Any]):
    device = torch.device(info["info"].get("device", "cpu"))
    dtype = info["info"].get("dtype", torch.float32)
    shape = info["info"].get("shape", [])
    mean = info["info"].get("mean", 0.0)
    std = info["info"].get("std", 1.0)
    min_val = info["info"].get("min_val")
    max_val = info["info"].get("max_val")
    if info.get("data") is not None:
        return info["data"].to(dtype=dtype, device=device)
    if dtype is torch.bool:
        return (torch.rand(shape, device=device) > 0.5)
    if std is None:
        std = 0.1
    if mean is None:
        mean = 0.0
    tensor = torch.randn(shape, device=device, dtype=dtype) * std * 0.2 + mean
    if min_val is not None:
        tensor = torch.clamp(tensor, min=min_val)
    if max_val is not None:
        tensor = torch.clamp(tensor, max=max_val)
    tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
    return tensor


def _get_dummy_tensor(info: Dict[str, Any]):
    device = torch.device(info["info"].get("device", "cpu"))
    dtype = info["info"].get("dtype", torch.float32)
    shape = info["info"].get("shape", [])
    if info.get("data") is not None:
        return info["data"].to(dtype=dtype, device=device)
    return torch.empty(shape, dtype=dtype, device=device)


def _modify_code_by_device(code: str, new_device_str: str):
    tree = ast.parse(code)

    class DeviceReplacer(ast.NodeTransformer):
        def __init__(self, new_device):
            super().__init__()
            self.new_device = new_device

        def visit_Call(self, node):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "device"
                and node.func.attr == "type"
            ):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    node.args[0].value = self.new_device
                return node

            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)
                and node.args[0].func.id == "device"
            ):
                device_call = node.args[0]
                for keyword in device_call.keywords:
                    if (
                        keyword.arg == "type"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        keyword.value.value = self.new_device
                return node

            new_keywords = []
            for keyword in node.keywords:
                if (
                    keyword.arg == "device"
                    and isinstance(keyword.value, ast.Call)
                    and isinstance(keyword.value.func, ast.Name)
                    and keyword.value.func.id == "device"
                ):
                    device_call = keyword.value
                    for kw in device_call.keywords:
                        if (
                            kw.arg == "type"
                            and isinstance(kw.value, ast.Constant)
                            and isinstance(kw.value.value, str)
                        ):
                            kw.value.value = self.new_device
                    new_keywords.append(keyword)
                else:
                    new_keywords.append(keyword)
            node.keywords = new_keywords
            return self.generic_visit(node)

    transformer = DeviceReplacer(new_device_str)
    modified_tree = transformer.visit(tree)
    ast.fix_missing_locations(modified_tree)
    return ast.unparse(modified_tree)


def _load_graph_module(model_path: str, target_device: str):
    source_path = os.path.join(model_path, "model.py")
    with open(source_path, "r", encoding="utf-8") as f:
        code = f.read()

    if target_device != "cuda":
        code = _modify_code_by_device(code, target_device)

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
        self.use_dummy_inputs = {{ use_dummy_inputs }}
        self.GraphModule = _load_graph_module(self.model_path, self.target_device)
        self.meta = _convert_meta_to_tensors(self.model_path)

    def _with_device(self, info: Dict[str, Any]):
        cloned = {"info": dict(info["info"]), "data": info.get("data")}
        cloned["info"]["device"] = self.target_device
        return cloned

    def test_forward_runs(self):
        model = self.GraphModule()
        weight_info = self.meta["weight_info"]

        def build_tensor(val):
            wrapped = self._with_device(val)
            return _get_dummy_tensor(wrapped) if self.use_dummy_inputs else _replay_tensor(wrapped)

        state_dict = {k: build_tensor(v) for k, v in weight_info.items()}
        model.__graph_net_file_path__ = self.model_path
        output = model(**state_dict)
        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
"""


class AgentUnittestGenerator:
    """Generate standalone unittest scripts for Torch samples."""

    def __init__(self, config: Dict[str, Any]):
        defaults = {
            "model_path": None,
            "output_path": None,
            "output_dir": None,
            "force_device": "auto",  # auto / cpu / cuda
            "use_dummy_inputs": False,
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
        self.use_dummy_inputs = merged["use_dummy_inputs"]

    def __call__(self, model):
        self.generate()
        return model

    def generate(self):
        output_path = self._resolve_output_path()
        target_device = self._choose_device()
        rendered = Template(TORCH_UNITTEST_TEMPLATE).render(
            target_device=target_device, use_dummy_inputs=self.use_dummy_inputs
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
        if self.force_device == "cuda":
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"


class AgentUnittestGeneratorPass(SamplePass):
    """SamplePass wrapper to generate Torch unittests via model_path_handler."""

    def __init__(self, config=None):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str = None,
        force_device: str = "auto",
        use_dummy_inputs: bool = False,
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
                "use_dummy_inputs": self.config["use_dummy_inputs"],
            }
        )
        generator.generate()
