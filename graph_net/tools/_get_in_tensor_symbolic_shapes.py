from pathlib import Path
from graph_net.dynamic_dim_constraints import DynamicDimConstraints
import sympy


class GetInTensorSymbolicShapes:
    def __init__(self, config):
        self.config = self.make_config(**config)

    def make_config(self, model_path_prefix):
        return {
            "model_path_prefix": model_path_prefix,
        }

    def __call__(self, model_path):
        original_model_path = Path(self.config["model_path_prefix"]) / model_path
        input_tensor_cstr_filepath = original_model_path / "input_tensor_constraints.py"
        if not input_tensor_cstr_filepath.exists():
            print(f"get-in-tensor-symbolic-shapes None {model_path}")
            return
        dyn_dim_cstrs = DynamicDimConstraints.unserialize_from_py_file(
            str(input_tensor_cstr_filepath)
        )
        dyn_dim_cstrs.symbol2example_value = {}
        dyn_dim_cstrs.input_shapes = sorted(
            [
                tuple(shape)
                for shape, name in dyn_dim_cstrs.input_shapes
                if any(isinstance(dim, sympy.Expr) for dim in shape)
            ],
            key=str,
        )
        input_shapes_str = str(dyn_dim_cstrs.input_shapes).replace(" ", "")
        print(f"get-in-tensor-symbolic-shapes {input_shapes_str} {model_path}")
