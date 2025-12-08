from pathlib import Path
from graph_net.dynamic_dim_constraints import DynamicDimConstraints


def get_dynamic_dim_constraints(model_path: str):
    original_model_path = Path(model_path)
    input_tensor_cstr_filepath = original_model_path / "input_tensor_constraints.py"
    if not input_tensor_cstr_filepath.exists():
        return None
    return DynamicDimConstraints.unserialize_from_py_file(
        str(input_tensor_cstr_filepath)
    )
