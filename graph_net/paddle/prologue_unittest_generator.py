import os
import re
import sys
import subprocess
import ast
import inspect
import jinja2
import textwrap
import tempfile
from pathlib import Path
from typing import Literal, List
from collections import namedtuple

import paddle
from athena.graphnet_samples import SubgraphGenerator
from graph_net import imp_util
from graph_net.paddle.extractor import GraphExtractor as BuiltinGraphExtractor
from graph_net.tensor_meta import TensorMeta


class GraphExtractor:
    def __init__(
        self,
        config: dict,
        model,
        name,
        dynamic,
        input_spec=None,
    ):
        self.model = model
        self.name = name.replace("/", "_")
        self.dynamic = dynamic
        self.input_spec = input_spec
        self.config = self.make_config(**config)

    def make_config(
        self,
        subgraph_range: list,
        device: Literal["auto", "cpu", "cuda", "xpu"] = "auto",
        try_run: bool = False,
        data_input_predicator_filepath: str = None,
        data_input_predicator_class_name: str = None,
        output_dir: str = "/tmp/prologue_unittests",
    ):
        assert isinstance(subgraph_range, (tuple, list)) and len(subgraph_range) == 2
        for pos in subgraph_range:
            assert isinstance(
                pos, int
            ), f"subgraph_range should be list of int, {subgraph_range=}"
        return {
            "subgraph_range": subgraph_range,
            "device": device,
            "try_run": try_run,
            "data_input_predicator_filepath": data_input_predicator_filepath,
            "data_input_predicator_class_name": data_input_predicator_class_name,
            "output_dir": output_dir,
        }

    def __call__(self, **input_dict):
        extracted_model = self.get_prologue_subgraph_unittest_generator()(**input_dict)
        return extracted_model

    def get_prologue_subgraph_unittest_generator(self):
        return PrologueSubgraphUnittestGenerator(
            config=self.config,
            parent_model=self.model,
            parent_model_name=self.name,
            parent_input_spec=self.input_spec,
        )


PADDLE_UNITTEST_TEMPLATE = r"""
import unittest
import numpy as np
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
    paddle.randint(low=0, high=2, shape={{shape}}, dtype='{{dtype}}').to(device='{{device}}')
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


class PrologueLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, {{graph_module_desc.prologue_weight_arg_names | join(", ")}}, {{graph_module_desc.prologue_input_arg_names | join(", ")}}):
        {{graph_module_desc.prologue_forward_body}}


class Model(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, {{graph_module_desc.weight_arg_names | join(", ")}}, {{graph_module_desc.input_arg_names | join(", ")}}):
        {{graph_module_desc.forward_body}}


def get_inputs():
    {%- for arg_name in graph_module_desc.input_arg_names %}
    {%- set input_idx = loop.index0 %}
    {{arg_name}} = {{get_input_tensor_instance(graph_module_desc.input_tensor_metas[input_idx], graph_module_desc.device)}}
    {%- endfor %}
    return [{{graph_module_desc.input_arg_names | join(", ")}}]


class {{graph_module_desc.model_name}}Test(unittest.TestCase):
    def setUp(self):
        paddle.seed(123)
        self.model = Model()

    def test_main(self):
        inputs = get_inputs()
        outputs = self.model(*inputs)


if __name__ == "__main__":
    unittest.main()
"""


GraphModuleDescriptor = namedtuple(
    "GraphModuleDescriptor",
    [
        "device",
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


class PrologueSubgraphUnittestGenerator:
    def __init__(
        self,
        config: dict,
        parent_model: paddle.nn.Layer,
        parent_model_name: str,
        parent_input_spec: List[paddle.static.InputSpec],
    ):
        self.config = config
        self.extracted = False
        self.parent_model_path = os.path.dirname(parent_model.__graph_net_file_path__)
        self.builtin_extractor = BuiltinGraphExtractor(
            model=parent_model,
            name=parent_model_name,
            dynamic=False,
            input_spec=parent_input_spec,
            workspace_path=self.config["output_dir"],
        )
        self.subgraph_range = self.config["subgraph_range"]
        self.device = self._choose_device(self.config["device"])
        self.try_run = self.config["try_run"]
        self.data_input_predicator = self._make_data_input_predicator(
            self.config["data_input_predicator_filepath"],
            self.config["data_input_predicator_class_name"],
        )

    def __call__(self, **input_dict):
        extracted_model = None
        if not self.extracted:
            extracted_model = self.do_extract(**input_dict)
            self.extracted = True
        return extracted_model

    def do_extract(self, **input_dict):
        # 1. Run the model to dump pir programs
        model_dump_path = os.path.join(
            self.builtin_extractor.dump_path, self.builtin_extractor.name
        )
        static_model = self.builtin_extractor.run_model_with_dump_enabled(
            model_dump_path, **input_dict
        )

        # 2. Convert pir programs to graphnet samples
        ir_programs_path = self.builtin_extractor.get_ir_programs_path(model_dump_path)
        example_inputs_path = self.builtin_extractor.get_example_inputs_path(
            model_dump_path
        )
        op_example_inputs_path = self.builtin_extractor.generate_op_example_inputs_path(
            model_dump_path, self.subgraph_range
        )
        generator = SubgraphGenerator(
            model_name=self.builtin_extractor.name,
            programs_file=ir_programs_path,
            example_inputs_file=example_inputs_path,
            op_example_inputs_file=op_example_inputs_path,
            eval_mode=True,
            tmp_dir=model_dump_path,
        )
        graphnet_sample_results = generator(self.subgraph_range, False)
        assert len(graphnet_sample_results) == 1

        return static_model

    def generate(self):
        print(f"[PrologueUnittestGenerator] Generate unittest for {self.model_path}")
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

        def _generate_unittest():
            graph_module_desc = GraphModuleDescriptor(
                device=self.device,
                model_name=model_name,
                input_arg_names=input_arg_names,
                input_tensor_metas=input_tensor_metas,
                weight_arg_names=weight_arg_names,
                weight_tensor_metas=weight_tensor_metas,
                forward_body=self._get_forward_body(
                    graph_module, input_arg_names, weight_arg_names
                ),
            )
            return self._render_template(graph_module_desc)

        # Generate unittest with main for try-run.
        unittest = _generate_unittest()
        self._write_to_file(unittest, self.output_dir)
        if self._try_to_run_unittest(unittest):
            self._write_to_file(unittest, self.output_dir)

    def _choose_device(self, device) -> str:
        import paddle

        if device in ["cpu", "gpu", "xpu"]:
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

    def _get_input_and_weight_arg_names(self, graph_module):
        input_arg_names = []
        weight_arg_names = []
        sig = inspect.signature(graph_module.forward)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            is_not_data_input = not self.data_input_predicator(self.model_path, name)
            if is_not_data_input:
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
        template_str = PADDLE_UNITTEST_TEMPLATE
        return jinja2.Template(template_str).render(graph_module_desc=graph_module_desc)
