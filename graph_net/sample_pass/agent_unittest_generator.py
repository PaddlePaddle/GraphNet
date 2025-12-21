import re
import sys
import subprocess
import ast
import inspect
import jinja2
import textwrap
import tempfile
from pathlib import Path
from typing import Literal
from collections import namedtuple

from graph_net import imp_util
from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
from graph_net.tensor_meta import TensorMeta


TORCH_UNITTEST_TEMPLATE = r"""
{%- if graph_module_desc.generate_main -%}
import unittest
{%- endif -%}
{{"\n"}}
import torch
from torch import device


{% macro get_input_tensor_instance(tensor_meta, device) -%}
{%- set shape = tensor_meta.shape -%}
{%- set dtype = tensor_meta.dtype -%}
{%- set data = tensor_meta.data -%}
{%- set min_val = tensor_meta.min_val -%}
{%- set max_val = tensor_meta.max_val -%}
{%- set mean = tensor_meta.mean -%}
{%- set std = tensor_meta.std -%}
{%- if data is not none -%}
    torch.tensor({{data}}, dtype={{dtype}}).reshape({{shape}}).to(device='{{device}}')
{%- elif dtype == "torch.bool" -%}
    torch.rand({{shape}}, device={{device}}) > 0.5
{%- elif dtype in ["torch.int8", "torch.int16", "torch.int32", "torch.int64"] -%}
    torch.randint({{min_val}}, {{max_val}} + 1, size={{shape}}, dtype={{dtype}}).to(device='{{device}}')
{%- elif dtype in ["torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"] -%}
    {%- if max_val is not none or min_val is not none -%}
    init_float_tensor(shape={{shape}}, dtype={{dtype}}, mean={{mean}}, std={{std}}, max_val={{max_val}}, min_val={{min_val}})
    {%- else -%}
    init_float_tensor(shape={{shape}}, dtype={{dtype}}, mean={{mean}}, std={{std}})
    {%- endif -%}
{%- endif -%}
{%- endmacro -%}


def init_float_tensor(shape, dtype, mean, std, max_val=None, min_val=None):
    tensor = torch.randn(size=shape) * std * 0.2 + mean
    if min_val is not None or max_val is not None:
        tensor = torch.clamp(tensor, min=min_val, max=max_val)
    return tensor.to(dtype).to('{{graph_module_desc.device}}')


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        {%- for arg_name in graph_module_desc.weight_arg_names %}
        {%- set input_idx = loop.index0 %}
        self.{{arg_name}} = {{get_input_tensor_instance(graph_module_desc.weight_tensor_metas[input_idx], graph_module_desc.device)}}
        {%- endfor %}

    def forward(self, {{graph_module_desc.input_arg_names | join(", ")}}):
        {{graph_module_desc.forward_body}}


def get_inputs():
    {%- for arg_name in graph_module_desc.input_arg_names %}
    {%- set input_idx = loop.index0 %}
    {{arg_name}} = {{get_input_tensor_instance(graph_module_desc.input_tensor_metas[input_idx], graph_module_desc.device)}}
    {%- endfor %}
    return [{{graph_module_desc.input_arg_names | join(", ")}}]


def get_init_inputs():
    return []

{{"\n"}}
{%- if graph_module_desc.generate_main -%}
class {{graph_module_desc.model_name}}Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)

        self.model = Model()

    def test_main(self):
        inputs = get_inputs()
        outputs = self.model(*inputs)


if __name__ == "__main__":
    unittest.main()
{%- endif -%}
"""


PADDLE_UNITTEST_TEMPLATE = r"""
{%- if graph_module_desc.generate_main -%}
import unittest
{%- endif -%}
{{"\n"}}
import paddle

{% macro get_input_tensor_instance(tensor_meta, device) -%}
{%- set shape = tensor_meta.shape -%}
{%- set dtype = tensor_meta.dtype -%}
{%- set data = tensor_meta.data -%}
{%- set min_val = tensor_meta.min_val -%}
{%- set max_val = tensor_meta.max_val -%}
{%- set mean = tensor_meta.mean -%}
{%- set std = tensor_meta.std -%}
{%- if data is not none -%}
    paddle.to_tensor({{data}}, dtype='{{dtype}}', shape={{shape}}).to(device='{{device}}')
{%- elif dtype == "bool" -%}
    paddle.randint(low=0, high=2, shape={{shape}}, dtype='{{dtype}}')
{%- elif dtype in ["int8", "int16", "int32", "int64"] -%}
    paddle.randint(low={{min_val}}, high={{max_val}} + 1, shape={{shape}}, dtype='{{dtype}}').to(device='{{device}}')
{%- elif dtype in ["float16", "bfloat16", "float32", "float64"] -%}
    {%- if mean is not none or std is not none -%}
    init_float_tensor(shape={{shape}}, dtype='{{dtype}}', max_val={{max_val}}, min_val={{min_val}}, mean={{mean}}, std={{std}})
    {%- else -%}
    init_float_tensor(shape={{shape}}, dtype='{{dtype}}', max_val={{max_val}}, min_val={{min_val}})
    {%- endif -%}
{%- endif -%}
{%- endmacro -%}


def init_float_tensor(shape, dtype, max_val, min_val, mean=None, std=None):
    if mean is not None and std is not None:
        tensor = paddle.randn(shape, dtype="float32") * std * 0.2 + mean
        tensor = paddle.clip(tensor, min=min_val, max=max_val)
    else:
        tensor = paddle.uniform(shape=shape, dtype="float32", min=min_val, max=max_val)
    return tensor.to(dtype).to('{{graph_module_desc.device}}')


class Model(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        {%- for arg_name in graph_module_desc.weight_arg_names %}
        {%- set input_idx = loop.index0 %}
        self.{{arg_name}} = {{get_input_tensor_instance(graph_module_desc.weight_tensor_metas[input_idx], graph_module_desc.device)}}
        {%- endfor %}

    def forward(self, {{graph_module_desc.input_arg_names | join(", ")}}):
        {{graph_module_desc.forward_body}}


def get_inputs():
    {%- for arg_name in graph_module_desc.input_arg_names %}
    {%- set input_idx = loop.index0 %}
    {{arg_name}} = {{get_input_tensor_instance(graph_module_desc.input_tensor_metas[input_idx], graph_module_desc.device)}}
    {%- endfor %}
    return [{{graph_module_desc.input_arg_names | join(", ")}}]


def get_init_inputs():
    return []

{{"\n"}}
{%- if graph_module_desc.generate_main -%}
class {{graph_module_desc.model_name}}Test(unittest.TestCase):
    def setUp(self):
        paddle.seed(123)
        self.model = Model()

    def test_main(self):
        inputs = get_inputs()
        outputs = self.model(*inputs)


if __name__ == "__main__":
    unittest.main()
{%- endif -%}
"""


GraphModuleDescriptor = namedtuple(
    "GraphModuleDescriptor",
    [
        "device",
        "generate_main",
        "model_name",
        "input_arg_names",
        "input_tensor_metas",
        "weight_arg_names",
        "weight_tensor_metas",
        "forward_body",
    ],
)


def load_class_from_file(file_path: str, class_name: str):
    print(f"Load {class_name} from {file_path}")
    module = imp_util.load_module(file_path, "unnamed")
    model_class = getattr(module, class_name, None)
    return model_class


class AgentUnittestGenerator:
    """Generate standalone unittest scripts for Torch samples."""

    def __init__(
        self,
        framework: str,
        model_path: str,
        output_name: str,
        output_dir: str,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        generate_main: bool = False,
        try_run: bool = False,
        data_input_predicator_filepath: str = None,
        data_input_predicator_class_name: str = None,
    ):
        self.framework = framework
        self.model_path = Path(model_path).resolve()
        self.output_name = output_name
        self.output_dir = Path(output_dir)
        self.device = self._choose_device(device)
        self.generate_main = generate_main
        self.try_run = try_run and generate_main
        self.data_input_predicator = self._make_data_input_predicator(
            data_input_predicator_filepath, data_input_predicator_class_name
        )

    def generate(self):
        print(f"[AgentUnittestGenerator] Generate unittest for {self.model_path}")
        model_name = "".join(
            word.capitalize() for word in re.split(r"[_.-]", self.model_path.name)
        )
        graph_module = load_class_from_file(
            self.model_path / "model.py", class_name="GraphModule"
        )
        input_arg_names, weight_arg_names = self._get_input_and_weight_arg_names(
            graph_module
        )
        (
            input_tensor_metas,
            weight_tensor_metas,
        ) = self._get_input_and_weight_tensor_metas(input_arg_names, weight_arg_names)
        graph_module_desc = GraphModuleDescriptor(
            device=self.device,
            generate_main=self.generate_main,
            model_name=model_name,
            input_arg_names=input_arg_names,
            input_tensor_metas=input_tensor_metas,
            weight_arg_names=weight_arg_names,
            weight_tensor_metas=weight_tensor_metas,
            forward_body=self._get_forward_body(
                graph_module, input_arg_names, weight_arg_names
            ),
        )
        unittest = self._render_template(graph_module_desc)
        if self._try_to_run_unittest(unittest):
            self._write_to_file(unittest, self.output_dir)

    def _choose_device(self, device) -> str:
        assert self.framework in ["torch", "paddle"], f"{self.framework=}"
        if self.framework == "torch":
            import torch

            if device in ["cpu", "cuda"]:
                return device
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif self.framework == "paddle":
            import paddle

            if device in ["cpu", "gpu"]:
                return device
            return "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"

    def _make_data_input_predicator(
        self, data_input_predicator_filepath, data_input_predicator_class_name
    ):
        if data_input_predicator_filepath and data_input_predicator_class_name:
            module = imp_util.load_module(data_input_predicator_filepath)
            cls = getattr(module, data_input_predicator_class_name)
            return cls(config={})
        return lambda *args, **kwargs: True

    def _write_to_file(self, unittest, output_dir):
        output_path = Path(output_dir) / self.output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(unittest, encoding="utf-8")
        print(
            f"[AgentUnittestGenerator] Generate unittest: {output_path} (device={self.device})"
        )
        return output_path

    def _try_to_run_unittest(self, unittest):
        if not self.try_run:
            return True

        with tempfile.TemporaryDirectory(prefix="unittest_") as temp_dir:
            output_path = self._write_to_file(unittest, temp_dir)
            result = subprocess.run(
                [sys.executable, output_path],
                check=True,
            )
            return result.returncode == 0

    def _is_parameter_type(self, annotation):
        if self.framework == "torch":
            import torch

            return annotation is torch.nn.parameter.Parameter
        elif self.framework == "paddle":
            import paddle

            return annotation is paddle.nn.parameter.Parameter

    def _get_input_and_weight_arg_names(self, graph_module):
        input_arg_names = []
        weight_arg_names = []
        sig = inspect.signature(graph_module.forward)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            is_not_data_input = not self.data_input_predicator(self.model_path, name)
            is_parameter_type = self._is_parameter_type(param.annotation)
            if is_not_data_input or is_parameter_type:
                weight_arg_names.append(name)
            else:
                input_arg_names.append(name)
        return input_arg_names, weight_arg_names

    def _get_input_and_weight_tensor_metas(self, input_arg_names, weight_arg_names):
        tensor_metas = TensorMeta.unserialize_from_py_file(
            self.model_path / "weight_meta.py"
        )
        tensor_metas.extend(
            TensorMeta.unserialize_from_py_file(self.model_path / "input_meta.py")
        )
        name2tensor_metas = {meta.name: meta for meta in tensor_metas}
        input_tensor_metas = [name2tensor_metas[name] for name in input_arg_names]
        weight_tensor_metas = [name2tensor_metas[name] for name in weight_arg_names]
        return input_tensor_metas, weight_tensor_metas

    def _get_forward_body(self, graph_module, input_arg_names, weight_arg_names):
        def _remove_clear_stmt_of_args(stmt):
            def _need_remove(target):
                return isinstance(target, ast.Name) and target.id in arg_names

            arg_names = input_arg_names + weight_arg_names
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is None
            ):
                # remove stmt like w_0 = None
                new_targets = [t for t in stmt.targets if not _need_remove(t)]
                if not new_targets:
                    return None
                stmt.targets = new_targets
            elif isinstance(stmt, ast.Delete):
                # remove stmt like del w_0
                new_targets = []
                for t in stmt.targets:
                    if isinstance(t, ast.Tuple):
                        kept = [e for e in t.elts if not _need_remove(e)]
                        if kept:
                            new_targets.append(ast.Tuple(elts=kept, ctx=ast.Del()))
                    elif not _need_remove(t):
                        new_targets.append(t)
                if not new_targets:
                    return None
                stmt.targets = new_targets
            return stmt

        def _rewrite_reference_for_weight(stmt):
            if isinstance(stmt, ast.Name):
                if isinstance(stmt.ctx, ast.Load) and stmt.id in weight_arg_names:
                    return ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr=stmt.id,
                        ctx=ast.Load(),
                    )
                return stmt

            for field, value in ast.iter_fields(stmt):
                if isinstance(value, list):
                    new_list = []
                    for item in value:
                        if isinstance(item, ast.AST):
                            item = _rewrite_reference_for_weight(item)
                        new_list.append(item)
                    setattr(stmt, field, new_list)
                elif isinstance(value, ast.AST):
                    setattr(stmt, field, _rewrite_reference_for_weight(value))
            return stmt

        def _update_for_weight(stmt):
            stmt = _remove_clear_stmt_of_args(stmt)
            if stmt is not None and weight_arg_names:
                stmt = _rewrite_reference_for_weight(stmt)
                ast.fix_missing_locations(stmt)
            return stmt

        source = inspect.getsource(graph_module.forward)
        lines = source.splitlines()
        num_indents = len(lines[-1]) - len(lines[-1].lstrip())

        tree = ast.parse(textwrap.dedent(source))
        func_def = tree.body[0]
        dedented_stmts = [
            ast.unparse(s)
            for stmt in func_def.body
            if (s := _update_for_weight(stmt)) is not None
        ]

        indent = " " * num_indents
        return f"\n{indent}".join(dedented_stmts)

    def _render_template(self, graph_module_desc):
        if self.framework == "torch":
            template_str = TORCH_UNITTEST_TEMPLATE
        elif self.framework == "paddle":
            template_str = PADDLE_UNITTEST_TEMPLATE
        return jinja2.Template(template_str).render(graph_module_desc=graph_module_desc)


class AgentUnittestGeneratorPass(SamplePass, ResumableSamplePassMixin):
    """SamplePass wrapper to generate Torch unittests via model_path_handler."""

    def __init__(self, config=None):
        super().__init__(config)

    def declare_config(
        self,
        framework: str,
        model_path_prefix: str,
        output_dir: str,
        device: str = "auto",
        generate_main: bool = False,
        try_run: bool = False,
        data_input_predicator_filepath: str = None,
        data_input_predicator_class_name: str = None,
        resume: bool = False,
        limits_handled_models: int = None,
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        dst_model_path = Path(self.config["output_dir"]) / rel_model_path
        if not dst_model_path.exists():
            return False
        output_name = self._get_output_name(rel_model_path)
        num_model_py_files = len(list(dst_model_path.rglob(output_name)))
        assert num_model_py_files <= 1
        return num_model_py_files == 1

    def _get_output_name(self, rel_model_path: str):
        return f"{Path(rel_model_path).name}_test.py"

    def resume(self, rel_model_path: str):
        model_path_prefix = Path(self.config["model_path_prefix"])
        output_dir = Path(self.config["output_dir"])
        generator = AgentUnittestGenerator(
            framework=self.config["framework"],
            model_path=str(model_path_prefix / rel_model_path),
            output_name=self._get_output_name(rel_model_path),
            output_dir=str(output_dir / rel_model_path),
            device=self.config["device"],
            generate_main=self.config["generate_main"],
            try_run=self.config["try_run"],
            data_input_predicator_filepath=self.config[
                "data_input_predicator_filepath"
            ],
            data_input_predicator_class_name=self.config[
                "data_input_predicator_class_name"
            ],
        )
        generator.generate()
