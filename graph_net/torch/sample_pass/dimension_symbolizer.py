import logging
from graph_net.sample_pass.sample_pass import SamplePass
from graph_net.sample_pass.resumable_sample_pass_mixin import ResumableSamplePassMixin
from graph_net.sample_pass.only_model_file_rewrite_sample_pass_mixin import (
    OnlyModelFileRewriteSamplePassMixin,
)
from graph_net.dynamic_dim_constraints import DynamicDimConstraints
from graph_net.imp_util import load_module
from graph_net.tensor_meta import TensorMeta
from graph_net.torch.static_to_dynamic import StaticToDynamic
import os
from contextlib import contextmanager
import tempfile
import shutil
from pathlib import Path
from dataclasses import asdict
import graph_net.graph_net_json_file_util as gn_json


class DimensionSymbolizer(
    SamplePass, ResumableSamplePassMixin, OnlyModelFileRewriteSamplePassMixin
):
    def __init__(self, config):
        super().__init__(config)

    def declare_config(
        self,
        model_path_prefix: str,
        output_dir: str,
        resume: bool = False,
        limits_handled_models: int = None,
        last_model_log_file: str = None,
    ):
        pass

    def __call__(self, rel_model_path: str):
        self.resumable_handle_sample(rel_model_path)

    def sample_handled(self, rel_model_path: str) -> bool:
        return self.naive_sample_handled(rel_model_path, search_file_name="model.py")

    def resume(self, rel_model_path: str):
        return self.copy_sample_and_handle_model_py_file(rel_model_path)

    def handle_model_py_file(self, rel_model_path: str) -> str:
        model_path = os.path.join(self.config["model_path_prefix"], rel_model_path)
        output_dir = Path(self.config["output_dir"])
        generalized_model_path = output_dir / rel_model_path
        generalized_model_path.mkdir(parents=True, exist_ok=True)
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
        assert len(dim_axes_pairs) > 0, f"No symbolic dims found. {model_path=}"

        def get_generalized():
            return self._get_generalized_model_py_file_path(
                dim_generalizer=dim_generalizer,
                dim_axes_pairs=dim_axes_pairs,
                model_path=model_path,
                inputs=inputs,
            )

        with get_generalized() as tmp_model_py_path:
            return Path(tmp_model_py_path).read_text()

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

    def _get_dimension_generalizer(self, dim_gen_pass_names):
        dim_generalizer = StaticToDynamic({"pass_names": dim_gen_pass_names})
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
