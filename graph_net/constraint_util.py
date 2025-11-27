from graph_net.dynamic_dim_constraints import DynamicDimConstraints
from contextlib import AbstractContextManager
from graph_net.imp_util import load_module
from graph_net.tensor_meta import TensorMeta
from typing import Callable
import functools
import copy
import sys
import os
from contextlib import contextmanager
import tempfile
import shutil
from pathlib import Path
import json
from dataclasses import asdict


class UpdateInputTensorConstraints:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = self._make_config(**config)
        self.data_input_predicator = self._make_data_input_predicator(self.config)
        self.model_runnable_predicator = self._make_model_runnable_predicator(
            self.config
        )
        self.num_successful_handled_models = 0

    def _make_data_input_predicator(self, config):
        module = load_module(config["data_input_predicator_filepath"])
        cls = getattr(module, config["data_input_predicator_class_name"])
        return cls(config["data_input_predicator_config"])

    def _make_model_runnable_predicator(self, config):
        module = load_module(config["model_runnable_predicator_filepath"])
        cls = getattr(module, config["model_runnable_predicator_class_name"])
        return cls(config["model_runnable_predicator_config"])

    def _make_config(
        self,
        data_input_predicator_filepath,
        model_runnable_predicator_filepath,
        data_input_predicator_class_name="DataInputPredicator",
        model_runnable_predicator_class_name="ModelRunner",
        data_input_predicator_config=None,
        model_runnable_predicator_config=None,
        dimension_generalizer_filepath=None,
        dimension_generalizer_class_name="StaticToDynamic",
        dimension_generalizer_config=None,
        model_path_prefix="",
        resume=False,
        last_model_log_file=None,
        limits_successfully_handled_models=None,
    ):
        if data_input_predicator_config is None:
            data_input_predicator_config = {}
        if model_runnable_predicator_config is None:
            model_runnable_predicator_config = {}
        if dimension_generalizer_config is None:
            dimension_generalizer_config = {}
        return {
            "resume": resume,
            "model_path_prefix": model_path_prefix,
            "data_input_predicator_filepath": data_input_predicator_filepath,
            "data_input_predicator_class_name": data_input_predicator_class_name,
            "data_input_predicator_config": data_input_predicator_config,
            "model_runnable_predicator_filepath": model_runnable_predicator_filepath,
            "model_runnable_predicator_class_name": model_runnable_predicator_class_name,
            "model_runnable_predicator_config": model_runnable_predicator_config,
            "dimension_generalizer_filepath": dimension_generalizer_filepath,
            "dimension_generalizer_class_name": dimension_generalizer_class_name,
            "dimension_generalizer_config": dimension_generalizer_config,
            "last_model_log_file": last_model_log_file,
            "limits_successfully_handled_models": limits_successfully_handled_models,
        }

    def __call__(self, model_path):
        model_path = os.path.join(self.config["model_path_prefix"], model_path)
        cstr_path = os.path.join(model_path, "input_tensor_constraints.py")
        if (
            self.config["resume"]
            and os.path.exists(cstr_path)
            and DynamicDimConstraints.kSymbols in open(cstr_path).read()
        ):
            module = load_module(cstr_path)
            symbols = getattr(module, DynamicDimConstraints.kSymbols)
            if len(symbols) > 0:
                return

        tensor_metas = self._get_tensor_metas(model_path)
        dyn_dim_cstr = make_dyn_dim_cstr_from_tensor_metas(tensor_metas)

        def data_input_predicator(input_var_name):
            return self.data_input_predicator(model_path, input_var_name)

        def get_tmp_model_path_ctx_mgr(dim_axes_pairs):
            return self._try_dimension_generalization(
                dim_axes_pairs, model_path, tensor_metas
            )

        def get_predicator_is_dyn_dim_cstr_feasible(tmp_model_path):
            def is_dyn_dim_cstr_feasible(dyn_dim_cstr):
                return self._is_dyn_dim_cstr_feasible(
                    tmp_model_path, tensor_metas, dyn_dim_cstr
                )

            return is_dyn_dim_cstr_feasible

        dyn_dim_cstr_feasibility_ctx_mgr = DynDimCstrFeasibilityContextManager(
            get_tmp_model_path_ctx_mgr=get_tmp_model_path_ctx_mgr,
            get_predicator_is_dyn_dim_cstr_feasible=get_predicator_is_dyn_dim_cstr_feasible,
        )
        dyn_dim_cstr, dim_gen_pass_names = symbolize_data_input_dims(
            dyn_dim_cstr,
            is_data_input=data_input_predicator,
            dyn_dim_cstr_feasibility_ctx_mgr=dyn_dim_cstr_feasibility_ctx_mgr,
        )
        self._save_dyn_dim_cstr(dyn_dim_cstr, model_path)
        self._save_dim_gen_pass_names(dim_gen_pass_names, model_path)
        if len(dyn_dim_cstr.symbols) > 0:
            self.num_successful_handled_models += 1
            limits = self.config["limits_successfully_handled_models"]
            if limits is not None:
                if self.num_successful_handled_models > limits:
                    print(
                        "`num_successful_handled_models` exceeds config `limits_successfully_handled_models`",
                        file=sys.stderr,
                    )
                    sys.exit(0)

    @contextmanager
    def _try_dimension_generalization(self, dim_axes_pairs, model_path, tensor_metas):
        if self.config["dimension_generalizer_filepath"] is None:
            yield model_path, ()
            return
        py_module = load_module(os.path.join(model_path, "model.py"))
        GraphModule = getattr(py_module, "GraphModule")
        GraphModule.__graph_net_file_path__ = py_module.__graph_net_file_path__
        model = GraphModule()
        decorator_cls = getattr(
            load_module(self.config["dimension_generalizer_filepath"]),
            self.config["dimension_generalizer_class_name"],
        )
        dim_generalizer = decorator_cls(self.config["dimension_generalizer_config"])
        dim_gen_pass = dim_generalizer(model, dim_axes_pairs)
        tensor_meta_attrs_list = [asdict(tensor_meta) for tensor_meta in tensor_metas]
        inputs = dim_gen_pass.create_inputs_by_metas(tensor_meta_attrs_list)
        if not dim_gen_pass.need_rewrite(inputs):
            yield model_path, ()
            return

        graph_module = dim_gen_pass.rewrite(inputs)
        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.copytree(Path(model_path), Path(tmp_dir), dirs_exist_ok=True)
            dim_gen_pass.save_graph_module(graph_module, tmp_dir)
            if self.config["last_model_log_file"] is not None:
                log_file = Path(self.config["last_model_log_file"])
                shutil.copy(Path(tmp_dir) / "model.py", log_file)
            yield tmp_dir, dim_gen_pass.get_pass_names()

    def _save_dim_gen_pass_names(self, dim_gen_pass_names, model_path):
        from graph_net.graph_net_json_file_util import kDimensionGeneralizationPasses

        graph_net_json_file_path = Path(f"{model_path}/graph_net.json")
        graph_net_json = json.loads(graph_net_json_file_path.read_text())
        graph_net_json[kDimensionGeneralizationPasses] = list(dim_gen_pass_names)
        graph_net_json_file_path.write_text(json.dumps(graph_net_json))

    def _save_dyn_dim_cstr(self, dyn_dim_cstr, model_path):
        cstr_code = dyn_dim_cstr.serialize_to_py_str()
        with open(os.path.join(model_path, "input_tensor_constraints.py"), "w") as fp:
            fp.write(cstr_code)

    def _get_tensor_metas(self, model_path):
        make = TensorMeta.unserialize_from_py_file
        return [
            *make(os.path.join(model_path, "input_meta.py")),
            *make(os.path.join(model_path, "weight_meta.py")),
        ]

    def _is_dyn_dim_cstr_feasible(
        self, model_path, tensor_metas, dyn_dim_cstr: DynamicDimConstraints
    ):
        tensor_metas = copy.deepcopy(tensor_metas)
        update_tensor_metas_by_dyn_dim_cstr(tensor_metas, dyn_dim_cstr)
        weight_meta_code = "\n".join(
            tensor_meta.serialize_to_py_str() for tensor_meta in tensor_metas
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            for filename in ["graph_net.json", "model.py"]:
                with open(os.path.join(tmpdir, filename), "w") as f:
                    f.write(open(os.path.join(model_path, filename)).read())
            with open(os.path.join(tmpdir, "input_meta.py"), "w") as f:
                f.write("")
            with open(os.path.join(tmpdir, "weight_meta.py"), "w") as f:
                f.write(weight_meta_code)
            return self.model_runnable_predicator(tmpdir)


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


def make_dyn_dim_cstr_from_tensor_metas(tensor_metas: list[TensorMeta]):
    named_shapes = [
        (shape, name)
        for tensor_meta in tensor_metas
        for name in [tensor_meta.name]
        for shape in [tensor_meta.shape]
    ]
    return DynamicDimConstraints.make_by_named_inputs(
        named_shapes=named_shapes,
    )


class DynDimCstrFeasibilityPredicator:
    def __init__(
        self,
        is_dyn_dim_cstr_feasible: Callable[[DynamicDimConstraints], bool],
        dim_gen_pass_names: tuple[str],
    ):
        self.is_dyn_dim_cstr_feasible = is_dyn_dim_cstr_feasible
        self.dim_gen_pass_names = dim_gen_pass_names

    def __call__(self, dyn_dim_cstr: DynamicDimConstraints) -> bool:
        return self.is_dyn_dim_cstr_feasible(dyn_dim_cstr)


class DynDimCstrFeasibilityContextManager:
    def __init__(
        self,
        get_tmp_model_path_ctx_mgr,
        get_predicator_is_dyn_dim_cstr_feasible,
    ):
        self.get_tmp_model_path_ctx_mgr = get_tmp_model_path_ctx_mgr
        self.get_predicator_is_dyn_dim_cstr_feasible = (
            get_predicator_is_dyn_dim_cstr_feasible
        )

    @contextmanager
    def __call__(
        self, dim_axes_pairs
    ) -> AbstractContextManager[DynDimCstrFeasibilityPredicator]:
        ctx_mgr = self.get_tmp_model_path_ctx_mgr
        with ctx_mgr(dim_axes_pairs) as (tmp_model_apth, dg_pass_names):
            predicator = self.get_predicator_is_dyn_dim_cstr_feasible(tmp_model_apth)
            yield DynDimCstrFeasibilityPredicator(predicator, dg_pass_names)


def symbolize_data_input_dims(
    dyn_dim_cstr: DynamicDimConstraints,
    is_data_input: Callable[[str], bool],
    dyn_dim_cstr_feasibility_ctx_mgr: DynDimCstrFeasibilityContextManager,
) -> (DynamicDimConstraints | None, tuple[str]):
    """
    is_data_input: Callable[["input_var_name:str"], bool]
    Symbolizes data input dimensions as much as possible.
    Returns new DynamicDimConstraints if success.
    Returns None if no symbolicable dim .
    """
    unqiue_dims = []
    dim2axes = {}

    def dumpy_filter_fn(input_name, input_idx, axis, dim):
        if is_data_input(input_name):
            print("data_input", input_name, input_idx, axis, dim)
            if dim not in unqiue_dims:
                unqiue_dims.append(dim)
                dim2axes[dim] = []
            dim2axes[dim].append(axis)
        # No symbolization by returning False
        return False

    # Collect input dimensions into `unqiue_dims`
    assert dyn_dim_cstr.symbolize(dumpy_filter_fn) is None
    total_dim_gen_pass_names = ()

    def append_dim_gen_pass_names(dim_gen_pass_names):
        nonlocal total_dim_gen_pass_names
        total_dim_gen_pass_names = tuple(
            [
                *total_dim_gen_pass_names,
                *(
                    pass_name
                    for pass_name in dim_gen_pass_names
                    if pass_name not in total_dim_gen_pass_names
                ),
            ]
        )

    for i, picked_dim in enumerate(unqiue_dims):
        cur_dyn_dim_cstr = copy.deepcopy(dyn_dim_cstr)

        def filter_fn(input_name, input_idx, axis, dim):
            return (
                is_data_input(input_name)
                and dim == picked_dim
                and (dim > 1 or axis == 0)
            )

        symbol = cur_dyn_dim_cstr.symbolize(filter_fn)
        if symbol is None:
            continue
        sym2example_value = {symbol: picked_dim + 1}
        if not cur_dyn_dim_cstr.check_delta_symbol2example_value(sym2example_value):
            continue
        dim_axes_pairs = tuple(
            (dim, axes) for dim in unqiue_dims[: i + 1] for axes in [dim2axes[dim]]
        )
        ctx_mgr = dyn_dim_cstr_feasibility_ctx_mgr
        with ctx_mgr(dim_axes_pairs) as dyn_dim_cstr_feasibility:
            tmp_dyn_dim_cstr = copy.deepcopy(cur_dyn_dim_cstr)
            tmp_dyn_dim_cstr.update_symbol2example_value(sym2example_value)
            if not dyn_dim_cstr_feasibility(tmp_dyn_dim_cstr):
                continue
            dyn_dim_cstr = cur_dyn_dim_cstr
            append_dim_gen_pass_names(dyn_dim_cstr_feasibility.dim_gen_pass_names)
    return dyn_dim_cstr, total_dim_gen_pass_names
