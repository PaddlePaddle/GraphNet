from graph_net.dynamic_dim_constraints import DynamicDimConstraints
from graph_net.imp_util import load_module
from graph_net.tensor_meta import TensorMeta
from typing import Callable
import functools
import copy
import sys
import os


class UpdateInputTensorConstraints:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = self._make_config(**config)
        self.data_input_predicator = self._make_data_input_predicator(self.config)
        self.model_runnable_predicator = self._make_model_runnable_predicator(
            self.config
        )

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
        data_input_predicator_config=None,
        model_runnable_predicator_class_name="ModelRunner",
        model_runnable_predicator_config=None,
        model_path_prefix="",
        resume=False,
    ):
        if data_input_predicator_config is None:
            data_input_predicator_config = {}
        if model_runnable_predicator_config is None:
            model_runnable_predicator_config = {}
        return {
            "data_input_predicator_filepath": data_input_predicator_filepath,
            "data_input_predicator_class_name": data_input_predicator_class_name,
            "data_input_predicator_config": data_input_predicator_config,
            "model_runnable_predicator_filepath": model_runnable_predicator_filepath,
            "model_runnable_predicator_class_name": model_runnable_predicator_class_name,
            "model_runnable_predicator_config": model_runnable_predicator_config,
            "model_path_prefix": model_path_prefix,
            "resume": resume,
        }

    def __call__(self, model_path):
        model_path = os.path.join(self.config["model_path_prefix"], model_path)
        print(f"{model_path=}")
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

        def is_dyn_dim_cstr_feasible(dyn_dim_cstr):
            return self._is_dyn_dim_cstr_feasible(
                model_path, tensor_metas, dyn_dim_cstr
            )

        dyn_dim_cstr = symbolize_data_input_dims(
            dyn_dim_cstr,
            is_data_input=data_input_predicator,
            is_dyn_dim_cstr_feasible=is_dyn_dim_cstr_feasible,
        )
        self._save_dyn_dim_cstr(dyn_dim_cstr, model_path)

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
        import tempfile

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


def symbolize_data_input_dims(
    dyn_dim_cstr: DynamicDimConstraints,
    is_data_input: Callable[[str], bool],
    is_dyn_dim_cstr_feasible: Callable[[DynamicDimConstraints], bool],
) -> DynamicDimConstraints | None:
    """
    is_data_input: Callable[["input_var_name:str"], bool]
    Symbolizes data input dimensions as much as possible.
    Returns new DynamicDimConstraints if success.
    Returns None if no symbolicable dim .
    """
    unqiue_dims = []

    def dumpy_filter_fn(input_name, input_idx, axis, dim):
        if is_data_input(input_name):
            print("data_input", input_name, input_idx, axis, dim)
            if dim not in unqiue_dims:
                unqiue_dims.append(dim)
        # No symbolization because of returning True
        return False

    # Collect input dimensions into `unqiue_dims`
    assert dyn_dim_cstr.symbolize(dumpy_filter_fn) is None
    for picked_dim in unqiue_dims:
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
        tmp_dyn_dim_cstr = copy.deepcopy(cur_dyn_dim_cstr)
        tmp_dyn_dim_cstr.update_symbol2example_value(sym2example_value)
        if not is_dyn_dim_cstr_feasible(tmp_dyn_dim_cstr):
            continue
        dyn_dim_cstr = cur_dyn_dim_cstr
    return dyn_dim_cstr
