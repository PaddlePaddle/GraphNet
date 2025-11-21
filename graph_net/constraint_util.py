from graph_net.dynamic_dim_constraints import DynamicDimConstraints
from typing import Callable
import copy


def symbolize_data_input_dims(
    dyn_dim_cstr: DynamicDimConstraints,
    is_data_input: Callable[["input_var_name:str"], bool],
    is_input_shape_valid: Callable[[DynamicDimConstraints], bool],
) -> DynamicDimConstraints | None:
    """
    Symbolizes data input dimensions as much as possible.
    Returns new DynamicDimConstraints if success.
    Returns None if no symbolicable dim .
    """
    unqiue_dims = set()

    def dumpy_filter_fn(input_name, input_idx, axis, dim):
        unqiue_dims.add(dim)
        # No symbolization because of returning True
        return False

    # Collect input dimensions into `unqiue_dims`
    assert dyn_dim_cstr.symbolize(dumpy_filter_fn) is None
    for picked_dim in unqiue_dims:
        tmp_dyn_dim_cstr = copy.deepcopy(dyn_dim_cstr)

        def filter_fn(input_name, input_idx, axis, dim):
            return is_data_input(input_name) and dim == picked_dim

        symbol = tmp_dyn_dim_cstr.symbolize(filter_fn)
        if symbol is None:
            continue
        sym2example_value = {symbol: picked_dim + 1}
        if not tmp_dyn_dim_cstr.check_delta_symbol2example_value(sym2example_value):
            continue
        tmp_dyn_dim_cstr.update_symbol2example_value(sym2example_value)
        if not is_input_shape_valid(tmp_dyn_dim_cstr):
            continue
        dyn_dim_cstr = tmp_dyn_dim_cstr
    return dyn_dim_cstr
