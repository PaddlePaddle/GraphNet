import copy


def get_all_symbol_names(constraint_attrs_list):
    unique_symbol_names = []
    for constraint_attrs in constraint_attrs_list:
        for dim in constraint_attrs["shape"]:
            if isinstance(dim, int):
                continue
            assert isinstance(dim, dict)
            if dim["symbol_name"] in unique_symbol_names:
                continue
            unique_symbol_names.append(dim["symbol_name"])

    return unique_symbol_names


def reify_symboli_dims(constraint_attrs_list, symbol_names):
    def try_reify_dim(dim):
        if isinstance(dim, int):
            return dim
        assert isinstance(dim, dict)
        if dim["symbol_name"] not in symbol_names:
            return dim
        return dim["example_value"]

    constraint_attrs_list = copy.deepcopy(constraint_attrs_list)
    for constraint_attrs in constraint_attrs_list:
        constraint_attrs["shape"] = [
            try_reify_dim(dim) for dim in constraint_attrs["shape"]
        ]
    return constraint_attrs_list


def modify_dim_example_value(constraint_attrs_list, symbol_name, modifier):
    def modify_dim(dim):
        if isinstance(dim, int):
            return
        assert isinstance(dim, dict)
        dim["example_value"] = modifier(dim["example_value"])

    constraint_attrs_list = copy.deepcopy(constraint_attrs_list)
    for constraint_attrs in constraint_attrs_list:
        for dim in constraint_attrs["shape"]:
            modify_dim(dim)
    return constraint_attrs_list


def symbolic_dims_all_reified(constraint_attrs_list):
    for constraint_attrs in constraint_attrs_list:
        for dim in constraint_attrs["shape"]:
            if not isinstance(dim, int):
                return False
    return True
