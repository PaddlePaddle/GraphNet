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
from graph_net.paddle.graph_meta_restorer import GraphMetaRestorer
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
        use_all_inputs: bool,
        device: Literal["auto", "cpu", "cuda", "xpu"] = "auto",
        tolerance: int = 0,
        try_run: bool = False,
        output_dir: str = "/tmp/prologue_unittests",
    ):
        assert isinstance(subgraph_range, (tuple, list)) and len(subgraph_range) == 2
        for pos in subgraph_range:
            assert isinstance(
                pos, int
            ), f"subgraph_range should be list of int, {subgraph_range=}"
        return {
            "subgraph_range": subgraph_range,
            "use_all_inputs": use_all_inputs,
            "device": device,
            "tolerance": tolerance,
            "try_run": try_run,
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
# This unittest is auto-generated from https://github.com/PaddlePaddle/GraphNet/tree/develop/{{graph_module_desc.rel_model_path_to_graphnet}}
# with subgraph_range={{graph_module_desc.subgraph_range}}.
#
# Usage:
# 1. Run following command on reference hardware.
#    python {{graph_module_desc.model_name}}_test.py --is-reference --device "cuda" --reference-dir "test_reference_outputs"
#
# 2. Copy all of the unittest and outputs of reference to target hardware.
#
# 3. Run following command on target hardware (xpu as example).
#    python {{graph_module_desc.model_name}}_test.py --device "xpu" --reference-dir "test_reference_outputs"
#

import os
import sys
import argparse
import unittest
import random
import numpy as np
import paddle


{% macro get_input_tensor_instance(tensor_meta) -%}
{%- set shape = tensor_meta.shape -%}
{%- set dtype = tensor_meta.dtype -%}
{%- set data = tensor_meta.data -%}
{%- set min_val = tensor_meta.min_val -%}
{%- set max_val = tensor_meta.max_val -%}
{%- set mean = tensor_meta.mean -%}
{%- set std = tensor_meta.std -%}
{%- if data is not none -%}
    paddle.reshape(paddle.to_tensor({{data}}, dtype='{{dtype}}'), shape={{shape}}).to(device=device)
{%- elif dtype == "bool" -%}
    init_integer_tensor(shape={{shape}}, dtype='{{dtype}}', device=device, min_val=0, max_val=1)
{%- elif dtype in ["int8", "int16", "int32", "int64"] -%}
    init_integer_tensor(shape={{shape}}, dtype='{{dtype}}', device=device, min_val={{min_val}}, max_val={{max_val}})
{%- elif dtype in ["float16", "bfloat16", "float32", "float64"] -%}
    {%- if mean is not none or std is not none -%}
    init_float_tensor(shape={{shape}}, dtype='{{dtype}}', device=device, min_val={{min_val}}, max_val={{max_val}}, mean={{mean}}, std={{std}})
    {%- else -%}
    init_float_tensor(shape={{shape}}, dtype='{{dtype}}', device=device, min_val={{min_val}}, max_val={{max_val}})
    {%- endif -%}
{%- endif -%}
{%- endmacro -%}


def init_integer_tensor(shape, dtype, device, min_val, max_val):
    array = np.random.randint(
        low=min_val, high=max_val + 1, size=shape, dtype="int64"
    )
    return paddle.to_tensor(array).to(dtype).to(device)


def init_float_tensor(shape, dtype, device, min_val, max_val, mean=None, std=None):
    if mean is not None and std is not None:
        array = np.random.normal(0, 1, shape) * std * 0.2 + mean
        array = np.clip(array, min_val, max_val)
    else:
        array = np.random.uniform(low=min_val, high=max_val, size=shape)
    return paddle.to_tensor(array).to(dtype).to(device)


class PrologueLayer(paddle.nn.Layer):
{{graph_module_desc.prologue_forward_func}}


class SuspectLayer(paddle.nn.Layer):
{{graph_module_desc.suspect_forward_func}}


class TestModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.prologue_layer = PrologueLayer()
        self.suspect_layer = SuspectLayer()

    def forward(self, {{graph_module_desc.arg_names | join(", ")}}):
        {{graph_module_desc.prologue_returns | join(", ")}} = self.prologue_layer({{graph_module_desc.prologue_arg_names | join(", ")}})
        {{graph_module_desc.suspect_returns | join(", ")}} = self.suspect_layer({{graph_module_desc.suspect_arg_names | join(", ")}})
        return ({{graph_module_desc.suspect_returns | join(", ")}},)


def get_input_dict(device):
    input_dict = {
        {%- for tensor_meta in graph_module_desc.tensor_metas %}
        '{{tensor_meta.name}}': {{get_input_tensor_instance(tensor_meta)}},
        {%- endfor %}
    }
    return input_dict


def tolerance_generator(tolerance, dtype):
    if dtype == paddle.float16:
        return 10 ** (tolerance * 3 / 5), 10**tolerance
    elif dtype == paddle.bfloat16:
        return 10 ** (tolerance * 1.796 / 5), 10**tolerance
    elif dtype == paddle.float32:
        return 10 ** (tolerance * 5.886 / 5), 10**tolerance
    elif dtype == paddle.float64:
        return 10 ** (tolerance * 7 / 5), 10 ** (tolerance * 7 / 5)
    else:
        return 0, 0


class {{graph_module_desc.test_name}}Test(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = TEST_ARGS.device
        self.is_reference = TEST_ARGS.is_reference
        self.reference_dir = TEST_ARGS.reference_dir
        self.tolerance = TEST_ARGS.tolerance
        self.random_seed = TEST_ARGS.random_seed
        self.runtime_seed = TEST_ARGS.runtime_seed

        paddle.seed(self.random_seed)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.input_dict = get_input_dict(self.device)
        self.test_model = TestModel()
        self.test_model.eval()

        if any(k in self.device for k in ["cuda", "gpu"]):
            paddle.set_flags({"FLAGS_cudnn_exhaustive_search": 1})

    def _flatten_outputs_to_list(self, outs):
        flattened_outs = outs
        if isinstance(outs, paddle.Tensor):
            flattened_outs = [outs]
        else:
            flattened_outs = [
                x
                for out in outs
                for x in (out if isinstance(out, (tuple, list)) else (out,))
            ]
        return flattened_outs

    def run_prologue_layer(self):
        prologue_inputs = [
            {%- for arg_name in graph_module_desc.prologue_arg_names %}
            self.input_dict['{{arg_name}}'],
            {%- endfor %}
        ]
        prologue_outputs = self.test_model.prologue_layer(*prologue_inputs)
        return self._flatten_outputs_to_list(prologue_outputs)

    def run_suspect_layer(self, prologue_outputs):
        suspect_inputs = [
        {%- for arg_name in graph_module_desc.suspect_arg_names %}
            {%- if arg_name not in graph_module_desc.prologue_returns %}
            self.input_dict['{{arg_name}}'],
            {%- else %}
            {%- for output_name in graph_module_desc.prologue_returns %}
                {%- if arg_name == output_name %}
            prologue_outputs[{{loop.index0}}],
                {%- endif %}
            {%- endfor %}
            {%- endif %}
        {%- endfor %}
        ]
        suspect_outputs = self.test_model.suspect_layer(*suspect_inputs)
        return self._flatten_outputs_to_list(suspect_outputs)

    def run_test_model(self):
        test_outputs = self.test_model(**self.input_dict)
        return self._flatten_outputs_to_list(test_outputs)

    def check_dtypes(self, reference_outputs, target_outputs):
        def _get_output_dtypes(outs):
            dtypes = [
                str(tensor.dtype).replace("paddle.", "")
                if isinstance(tensor, paddle.Tensor)
                else None
                for i, tensor in enumerate(outs)
            ]
            return dtypes

        reference_dtypes = _get_output_dtypes(reference_outputs)
        target_dtypes = _get_output_dtypes(target_outputs)
        dtype_match = all(
            reference == target for reference, target in zip(reference_dtypes, target_dtypes)
        )
        self.assertTrue(dtype_match, f"Data type of outputs are not matched ({reference_dtypes=} vs {target_dtypes=}).")

    def check_shapes(self, reference_outputs, target_outputs):
        def _get_output_shapes(outs):
            shapes = [
                tensor.shape if isinstance(tensor, paddle.Tensor) else None
                for i, tensor in enumerate(outs)
            ]
            return shapes

        reference_shapes = _get_output_shapes(reference_outputs)
        target_shapes = _get_output_shapes(target_outputs)
        shape_match = all(
            reference == target for reference, target in zip(reference_shapes, target_shapes)
        )
        self.assertTrue(shape_match, f"Shape of outputs are not matched ({reference_shapes=} vs {target_shapes=}).")

    def check_results(self, reference_outputs, target_outputs):
        def _convert_to_numpy(out):
            if out.dtype not in [paddle.float32, paddle.float64]:
                return out.cast("float32").numpy()
            else:
                return out.numpy()

        assert len(reference_outputs) == len(target_outputs), f"The number of outputs is not equal ({len(reference_outputs)=} vs {len(target_outputs)})."
        self.check_dtypes(reference_outputs, target_outputs)
        self.check_shapes(reference_outputs, target_outputs)

        for reference, target in zip(reference_outputs, target_outputs):
            atol, rtol = tolerance_generator(self.tolerance, reference.dtype)
            np.testing.assert_allclose(
                actual=_convert_to_numpy(target),
                desired=_convert_to_numpy(reference),
                atol=atol,
                rtol=rtol,
            )

    def test_separated(self):
        paddle.seed(self.runtime_seed)
        prologue_output_path = os.path.join(self.reference_dir, "{{graph_module_desc.model_name}}_prologue_reference.pdout")
        prologue_outputs = self.run_prologue_layer()
        if self.is_reference:
            print(f"Save prologue output tensors to {prologue_output_path}.")
            paddle.save(prologue_outputs, prologue_output_path)
            prologue_reference_outputs = prologue_outputs
        else:
            print(f"Load prologue output tensors from {prologue_output_path}")
            prologue_reference_outputs = paddle.load(prologue_output_path)
            with self.subTest(name="check_prologue_outputs"):
                self.check_results(prologue_reference_outputs, prologue_outputs)

        test_output_path = os.path.join(self.reference_dir, "{{graph_module_desc.model_name}}_separated_reference.pdout")
        test_outputs = self.run_suspect_layer(prologue_reference_outputs)
        if self.is_reference:
            print(f"Save test output tensors to {test_output_path}.")
            paddle.save(test_outputs, test_output_path)
        else:
            print(f"Load test output tensors on reference device from {test_output_path}.")
            test_reference_outputs = paddle.load(test_output_path)
            with self.subTest(name="check_suspect_outputs"):
                self.check_results(test_reference_outputs, test_outputs)

    def test_combined(self):
        paddle.seed(self.runtime_seed)
        test_output_path = os.path.join(self.reference_dir, "{{graph_module_desc.model_name}}_combined_reference.pdout")
        test_outputs = self.run_test_model()
        if self.is_reference:
            print(f"Save test output tensors to {test_output_path}.")
            paddle.save(test_outputs, test_output_path)
        else:
            print(f"Load test output tensors on reference device from {test_output_path}.")
            test_reference_outputs = paddle.load(test_output_path)
            with self.subTest(name="check_combined_outputs"):
                self.check_results(test_reference_outputs, test_outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is-reference", action="store_true", default=False, help="Whether it runs on the reference hardware.")
    parser.add_argument("--device", type=str, required=True, help="Device to run on.")
    parser.add_argument("--reference-dir", type=str, required=True, help="Directory to save the results on reference hardware.")
    parser.add_argument("--tolerance", type=int, default={{graph_module_desc.tolerance}}, help="Tolerance level used in allclose check.")
    parser.add_argument("--random-seed", type=int, default=123, help="Random seed to initialize the input tensors.")
    parser.add_argument("--runtime-seed", type=int, default=1024, help="Random seed for runtime.")
    args, remaining = parser.parse_known_args()

    global TEST_ARGS
    TEST_ARGS = args

    print(f"PaddlePaddle version: {paddle.__version__}")
    print(f"PaddlePaddle commit: {paddle.version.commit}")
    unittest.main(argv=[sys.argv[0]] + remaining)
"""


GraphModuleDescriptor = namedtuple(
    "GraphModuleDescriptor",
    [
        "model_name",
        "test_name",
        "tolerance",
        "arg_names",
        "tensor_metas",
        "prologue_arg_names",
        "prologue_returns",
        "prologue_forward_func",
        "suspect_arg_names",
        "suspect_returns",
        "suspect_forward_func",
        "rel_model_path_to_graphnet",
        "subgraph_range",
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
        self.model_name = parent_model_name
        self.parent_model_path = os.path.dirname(parent_model.__graph_net_file_path__)
        self.builtin_extractor = BuiltinGraphExtractor(
            model=parent_model,
            name=parent_model_name,
            dynamic=False,
            input_spec=parent_input_spec,
            workspace_path=self.config["output_dir"],
        )
        self.subgraph_range = self.config["subgraph_range"]
        self.use_all_inputs = self.config["use_all_inputs"]
        self.device = self._choose_device(self.config["device"])
        self.tolerance = self.config["tolerance"]
        self.try_run = self.config["try_run"]
        self.output_dir = self.config["output_dir"]
        self.graph_meta_restorer = self._make_graph_meta_restorer()

    def __call__(self, **input_dict):
        extracted_model = None
        if not self.extracted:
            extracted_model = self.do_extract(**input_dict)
            self.extracted = True
        return extracted_model

    def do_extract(self, **input_dict):
        print(
            f"[PrologueUnittestGenerator] Generate unittest for {self.parent_model_path} with subgraph_range={self.subgraph_range}"
        )

        # 1. Run the model to dump pir programs
        model_dump_path = os.path.join(
            self.builtin_extractor.dump_path, self.model_name
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
        subgraph_generator = SubgraphGenerator(
            model_name=self.model_name,
            programs_file=ir_programs_path,
            example_inputs_file=example_inputs_path,
            op_example_inputs_file=op_example_inputs_path,
            eval_mode=True,
            tmp_dir=model_dump_path,
        )

        # 3. Generate unittest
        with tempfile.TemporaryDirectory(prefix="prologue_unittest_") as tmp_dir:
            self.generate(subgraph_generator, tmp_dir)
        return static_model

    def _save_and_get_graph_module(
        self, subgraph_generator, subgraph_range, use_all_inputs, tmp_dir
    ):
        results = subgraph_generator(subgraph_range, False, use_all_inputs)
        assert len(results) == 1
        output_name = f"{subgraph_range[0]}_{subgraph_range[1]}"
        output_path = os.path.join(tmp_dir, f"{self.model_name}-{output_name}")
        self.builtin_extractor.write_sample_to_file(output_path, results[0])
        graph_module = load_class_from_file(
            Path(output_path) / "model.py", class_name="GraphModule"
        )
        return graph_module, output_path

    def generate(self, subgraph_generator, tmp_dir):
        test_name = "".join(
            word.capitalize() for word in re.split(r"[_.-]", self.model_name)
        )

        graph_module, output_path = self._save_and_get_graph_module(
            subgraph_generator, self.subgraph_range, self.use_all_inputs, tmp_dir
        )
        arg_names = self._get_forward_arg_names(graph_module)
        self.graph_meta_restorer(output_path, self.subgraph_range, self.use_all_inputs)
        tensor_metas = self._get_tensor_metas(output_path)

        # prologue model information
        prologue_subgraph_range = [self.subgraph_range[0], self.subgraph_range[1] - 1]
        prologue_graph_module, _ = self._save_and_get_graph_module(
            subgraph_generator, prologue_subgraph_range, False, tmp_dir
        )
        prologue_forward_func, prologue_returns = self._get_forward_func_and_returns(
            prologue_graph_module
        )
        prologue_arg_names = self._get_forward_arg_names(prologue_graph_module)

        # suspect model information
        suspect_subgraph_range = [self.subgraph_range[1] - 1, self.subgraph_range[1]]
        suspect_graph_module, _ = self._save_and_get_graph_module(
            subgraph_generator, suspect_subgraph_range, False, tmp_dir
        )
        suspect_forward_func, suspect_returns = self._get_forward_func_and_returns(
            suspect_graph_module
        )
        suspect_arg_names = self._get_forward_arg_names(suspect_graph_module)

        def _generate_unittest():
            graph_module_desc = GraphModuleDescriptor(
                model_name=self.model_name,
                test_name=test_name,
                tolerance=self.tolerance,
                arg_names=arg_names,
                tensor_metas=tensor_metas,
                prologue_arg_names=prologue_arg_names,
                prologue_returns=prologue_returns,
                prologue_forward_func=prologue_forward_func,
                suspect_arg_names=suspect_arg_names,
                suspect_returns=suspect_returns,
                suspect_forward_func=suspect_forward_func,
                rel_model_path_to_graphnet=self.parent_model_path.split("GraphNet/")[
                    -1
                ],
                subgraph_range=self.subgraph_range,
            )
            return self._render_template(graph_module_desc)

        # Generate unittest with main for try-run.
        unittest = _generate_unittest()
        self._write_to_file(unittest, self.output_dir)
        if self._try_to_run_unittest(unittest, tmp_dir):
            self._write_to_file(unittest, self.output_dir)

    def _choose_device(self, device) -> str:
        if device in ["cpu", "gpu", "xpu"]:
            return device
        return paddle.get_device()

    def _make_graph_meta_restorer(self):
        config = {
            "update_inplace": True,
            "input_meta_allow_partial_update": False,
        }
        graph_meta_restorer = GraphMetaRestorer(
            config=config, parent_model_path=self.parent_model_path
        )
        return graph_meta_restorer

    def _write_to_file(self, unittest, output_dir):
        output_path = Path(output_dir) / f"{self.model_name}_test.py"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(unittest, encoding="utf-8")
        print(f"[PrologueUnittestGenerator] Generate unittest: {output_path}")
        return output_path

    def _try_to_run_unittest(self, unittest, tmp_dir):
        if not self.try_run:
            return True

        output_path = self._write_to_file(unittest, tmp_dir)
        result = subprocess.run(
            [
                sys.executable,
                output_path,
                "--is-reference",
                "--device",
                self.device,
                "--reference-dir",
                os.path.join(tmp_dir, "test_reference_outputs"),
            ],
            check=True,
        )
        return result.returncode == 0

    def _get_tensor_metas(self, model_path):
        tensor_metas = TensorMeta.unserialize_from_py_file_order_preserved(
            Path(model_path) / "weight_meta.py"
        )
        tensor_metas.extend(
            TensorMeta.unserialize_from_py_file_order_preserved(
                Path(model_path) / "input_meta.py"
            )
        )
        for meta in tensor_metas:
            if meta.min_val is None:
                meta.min_val = 0
            if meta.max_val is None:
                meta.max_val = 2
        return tensor_metas

    def _get_forward_arg_names(self, graph_module):
        arg_names = []
        sig = inspect.signature(graph_module.forward)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            arg_names.append(name)
        return arg_names

    def _get_forward_func_and_returns(self, graph_module):
        # the whole definition of forward function
        source = inspect.getsource(graph_module.forward)
        lines = source.splitlines()
        num_indents = len(lines[0]) - len(lines[0].lstrip())

        tree = ast.parse(textwrap.dedent(source))
        func_def = tree.body[0]
        forward_func = textwrap.indent(ast.unparse(func_def), " " * num_indents)

        # the return statements
        return_node_values = [
            node.value for node in ast.walk(func_def) if isinstance(node, ast.Return)
        ]
        assert len(return_node_values) == 1 and return_node_values[0] is not None
        return_code = ast.unparse(return_node_values[0])
        return_names = [name.strip() for name in return_code.strip("()").split(",")]
        return forward_func, return_names

    def _render_template(self, graph_module_desc):
        template_str = PADDLE_UNITTEST_TEMPLATE
        return jinja2.Template(template_str).render(graph_module_desc=graph_module_desc)
