import logging
from graph_net.dynamic_dim_constraints import DynamicDimConstraints
from graph_net.imp_util import load_module
from graph_net.tensor_meta import TensorMeta
import functools
import sys
import os
from contextlib import contextmanager
import tempfile
import shutil
from pathlib import Path
from dataclasses import asdict
import graph_net.graph_net_json_file_util as gn_json


class ApplyDimGenPasses:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = self._make_config(**config)
        self.num_handled_models = 0

    def _make_config(
        self,
        output_dir: str,
        dimension_generalizer_filepath=None,
        dimension_generalizer_class_name="StaticToDynamic",
        dimension_generalizer_config=None,
        model_path_prefix="",
        resume=False,
        last_model_log_file=None,
        limits_handled_models=None,
    ):
        if dimension_generalizer_config is None:
            dimension_generalizer_config = {}
        return {
            "resume": resume,
            "output_dir": output_dir,
            "model_path_prefix": model_path_prefix,
            "dimension_generalizer_filepath": dimension_generalizer_filepath,
            "dimension_generalizer_class_name": dimension_generalizer_class_name,
            "dimension_generalizer_config": dimension_generalizer_config,
            "last_model_log_file": last_model_log_file,
            "limits_handled_models": limits_handled_models,
        }

    def __call__(self, rel_model_path):
        model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        generalized_model_path = output_dir / rel_model_path
        if self.config["resume"] and (generalized_model_path / "model.py").exists():
            return
        tensor_metas = self._get_tensor_metas(model_path)
        tensor_meta_attrs_list = [asdict(tensor_meta) for tensor_meta in tensor_metas]
        dim_gen_pass_names = self._get_dim_gen_pass_names(model_path)
        dim_generalizer = self._get_dimension_generalizer(dim_gen_pass_names)
        inputs = dim_generalizer.create_inputs_by_metas(
            module=self._get_model(model_path),
            tensor_meta_attrs_list=tensor_meta_attrs_list,
        )
        dyn_dim_cstrs = DynamicDimConstraints.unserialize_from_py_file(
            os.path.join(model_path, "input_tensor_constraints.py")
        )
        dim_axes_pairs = self._get_dim_axes_pairs(dyn_dim_cstrs)
        if len(dim_axes_pairs) == 0:
            return

        def get_generalized():
            return self._get_generalized_model_py_file_path(
                dim_generalizer=dim_generalizer,
                dim_axes_pairs=dim_axes_pairs,
                model_path=model_path,
                inputs=inputs,
            )

        with get_generalized() as generalized_model_py_path:
            self._save_generalized_model_path(rel_model_path, generalized_model_py_path)

        self._check_num_handled_models()

    def _save_generalized_model_path(self, rel_model_path, generalized_model_py_path):
        from_model_path = Path(self.config["model_path_prefix"]) / rel_model_path
        to_model_path = Path(self.config["output_dir"]) / rel_model_path
        print(f"{str(to_model_path)=}")
        to_model_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(from_model_path), Path(to_model_path), dirs_exist_ok=True)
        generalized_model_py_code = Path(generalized_model_py_path).read_text()
        (to_model_path / "model.py").write_text(generalized_model_py_code)

    def _get_dim_axes_pairs(self, dyn_dim_cstrs):
        sym_input_shapes = dyn_dim_cstrs.get_sorted_symbolic_input_shapes()
        return [
            (dim, axes)
            for symbol in dyn_dim_cstrs.symbols
            for dim in [dyn_dim_cstrs.symbol2example_value[symbol]]
            for axes in [
                [
                    axis
                    for shape in sym_input_shapes
                    for axis, sym_or_dim in enumerate(shape)
                    if sym_or_dim == symbol
                ]
            ]
        ]

    def _get_dim_gen_pass_names(self, model_path):
        json_value = gn_json.read_json(model_path)
        return json_value.get(gn_json.kDimensionGeneralizationPasses, [])

    def _check_num_handled_models(self):
        self.num_handled_models += 1
        limits = self.config["limits_handled_models"]
        if limits is None:
            return
        if self.num_handled_models < limits:
            return
        print("`num_handled_models` exceeds config `limits_handled_models`")
        sys.exit(0)

    def _get_dimension_generalizer(self, dim_gen_pass_names):
        assert self.config["dimension_generalizer_filepath"] is not None
        decorator_cls = getattr(
            load_module(self.config["dimension_generalizer_filepath"]),
            self.config["dimension_generalizer_class_name"],
        )
        config = {"pass_names": dim_gen_pass_names}
        dim_generalizer = decorator_cls(config)
        return dim_generalizer

    def _get_model(self, model_path):
        py_module = load_module(os.path.join(model_path, "model.py"))
        GraphModule = getattr(py_module, "GraphModule")
        GraphModule.__graph_net_file_path__ = py_module.__graph_net_file_path__
        return GraphModule()

    @contextmanager
    def _get_generalized_model_py_file_path(
        self, dim_generalizer, dim_axes_pairs, model_path, inputs
    ):
        model = self._get_model(model_path)
        dim_gen_pass = dim_generalizer(model, dim_axes_pairs)
        logging.warning("before need_rewrite")
        need_rewrite = dim_gen_pass.need_rewrite(inputs)
        logging.warning("after need_rewrite")
        if not need_rewrite:
            yield os.path.join(model_path, "model.py")
            return
        logging.warning("before rewrite")
        graph_module = dim_gen_pass.rewrite(inputs)
        logging.warning("after rewrite")
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copytree(Path(model_path), Path(tmp_dir), dirs_exist_ok=True)
            dim_gen_pass.save_graph_module(graph_module, tmp_dir)
            yield os.path.join(tmp_dir, "model.py")

    def _get_tensor_metas(self, model_path):
        make = TensorMeta.unserialize_from_py_file
        return [
            *make(os.path.join(model_path, "input_meta.py")),
            *make(os.path.join(model_path, "weight_meta.py")),
        ]


def update_tensor_metas_by_dyn_dim_cstr(
    tensor_metas: list[TensorMeta], dyn_dim_cstr: DynamicDimConstraints
):
    input_shapes = dyn_dim_cstr.get_reified_input_shapes()
    assert len(tensor_metas) == len(input_shapes)
    for i, tensor_meta in enumerate(tensor_metas):
        tensor_meta.shape = input_shapes[i]
        if tensor_meta.data is not None:
            assert isinstance(tensor_meta.data, (list, tuple))
            size = functools.reduce(lambda a, b: a * b, tensor_meta.shape, 1)
            doubled_data = [*tensor_meta.data, *tensor_meta.data]
            tensor_meta.data = doubled_data[:size]
