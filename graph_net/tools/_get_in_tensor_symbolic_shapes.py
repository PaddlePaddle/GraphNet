import sympy
from pathlib import Path
from graph_net.dynamic_dim_constraints import DynamicDimConstraints
import graph_net.graph_net_json_file_util as gn_json


class GetInTensorSymbolicShapes:
    def __init__(self, config):
        self.config = self.make_config(**config)

    def make_config(self, model_path_prefix, ignore_reified=True):
        return {
            "model_path_prefix": model_path_prefix,
            "ignore_reified": ignore_reified,
        }

    def __call__(self, model_path):
        original_model_path = Path(self.config["model_path_prefix"]) / model_path
        input_tensor_cstr_filepath = original_model_path / "input_tensor_constraints.py"
        if not input_tensor_cstr_filepath.exists():
            print(f"get-in-tensor-symbolic-shapes None {model_path}")
            return
        if self.config["ignore_reified"] and self._found_reified_dims(
            str(original_model_path)
        ):
            print(f"get-in-tensor-symbolic-shapes <reified> {model_path}")
            return
        dyn_dim_cstrs = DynamicDimConstraints.unserialize_from_py_file(
            str(input_tensor_cstr_filepath)
        )
        for shape, name in dyn_dim_cstrs.input_shapes:
            if not any(isinstance(dim, sympy.Expr) for dim in shape):
                continue
            print(f"{shape=} {name=}")
        input_shapes_str = str(dyn_dim_cstrs.serialize_symbolic_input_shapes_to_str())
        print(f"get-in-tensor-symbolic-shapes {input_shapes_str} {model_path}")

    def _found_reified_dims(self, model_path):
        json = gn_json.read_json(model_path)

        if gn_json.kSymbolicDimensionReifier not in json:
            return False

        return json[gn_json.kSymbolicDimensionReifier] is not None
