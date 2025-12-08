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
from collections import OrderedDict
import copy
from graph_net.hash_util import get_sha256_hash


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
        if (
            self.config["resume"]
            and generalized_model_path.exists()
            and generalized_model_path.is_dir()
            and len(list(generalized_model_path.iterdir())) > 0
        ):
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
            print("No symbolic dims found. {model_path=}")
            return

        def get_generalized():
            return self._get_generalized_model_py_file_path(
                dim_generalizer=dim_generalizer,
                dim_axes_pairs=dim_axes_pairs,
                model_path=model_path,
                inputs=inputs,
            )

        with get_generalized() as tmp_model_py_path:
            from_model_path = Path(self.config["model_path_prefix"]) / rel_model_path
            triples = self._get_reified_tensor_metas(from_model_path, dyn_dim_cstrs)
            for symbol2example_value, cur_tensor_metas, cur_dyn_dim_cstrs in triples:
                to_model_path = self._get_to_model_path(
                    rel_model_path, symbol2example_value
                )
                print(f"{str(to_model_path)=}")
                self._copy_sample_model_path(from_model_path, to_model_path)
                self._save_generalized_model_path(to_model_path, tmp_model_py_path)
                self._save_tensor_metas_as_weight_meta(to_model_path, cur_tensor_metas)
                self._save_dyn_dim_cstrs(to_model_path, cur_dyn_dim_cstrs)

        self._check_num_handled_models()

    def _get_reified_tensor_metas(self, from_model_path, dyn_dim_cstrs):
        tensor_metas = self._get_tensor_metas(str(from_model_path))
        symbols, reified_dims = self._get_symbols_and_reified_dims(
            from_model_path, dyn_dim_cstrs
        )
        for dims in reified_dims:
            symbol2example_value = OrderedDict(list(zip(symbols, dims)))
            cur_dyn_dim_cstrs = copy.deepcopy(dyn_dim_cstrs)
            cur_tensor_metas = copy.deepcopy(tensor_metas)
            cur_dyn_dim_cstrs.update_symbol2example_value(symbol2example_value)
            update_tensor_metas_by_dyn_dim_cstr(cur_tensor_metas, cur_dyn_dim_cstrs)
            yield symbol2example_value, cur_tensor_metas, cur_dyn_dim_cstrs

    def _get_symbols_and_reified_dims(self, from_model_path, dyn_dim_cstrs):
        json_value = gn_json.read_json(str(from_model_path))
        reifier_name = json_value[gn_json.kSymbolicDimensionReifier]
        from graph_net.torch.sym_dim_reifiers.reifier_mgr import get_reifier

        reifier_class = get_reifier(reifier_name)
        reifier_instance = reifier_class(str(from_model_path))
        assert reifier_instance.match
        symbols2reified_dims = reifier_instance.reify()
        assert len(symbols2reified_dims) == 1
        symbols, reified_dims = next(iter(symbols2reified_dims.items()))
        assert tuple(symbols) == tuple(dyn_dim_cstrs.symbols)
        assert all(len(symbols) == len(dims) for dims in reified_dims)
        return symbols, reified_dims

    def _save_dyn_dim_cstrs(self, to_model_path, dyn_dim_cstrs):
        cstr_code = dyn_dim_cstrs.serialize_to_py_str()
        (to_model_path / "input_tensor_constraints.py").write_text(cstr_code)

    def _save_tensor_metas_as_weight_meta(self, to_model_path, tensor_metas):
        weight_meta_code = "\n".join(
            tensor_meta.serialize_to_py_str() for tensor_meta in tensor_metas
        )
        (to_model_path / "weight_meta.py").write_text(weight_meta_code)

    def _get_to_model_path(self, rel_model_path, symbol2example_value):
        sym_dim_str = "_".join(
            f"{sym_name}_{dim}"
            for symbol, dim in symbol2example_value.items()
            for sym_name in [symbol.name]
        )
        sub_module_name = f"{os.path.basename(rel_model_path)}__{sym_dim_str}"
        to_model_path = (
            Path(self.config["output_dir"]) / rel_model_path / sub_module_name
        )
        return to_model_path

    def _copy_sample_model_path(self, from_model_path, to_model_path):
        to_model_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(from_model_path), Path(to_model_path), dirs_exist_ok=True)

    def _save_generalized_model_path(self, to_model_path, tmp_model_py_path):
        generalized_model_py_code = Path(tmp_model_py_path).read_text()
        (to_model_path / "model.py").write_text(generalized_model_py_code)
        file_hash = get_sha256_hash(generalized_model_py_code)
        (to_model_path / "graph_hash.txt").write_text(file_hash)

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
