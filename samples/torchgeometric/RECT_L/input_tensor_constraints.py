from sympy import Symbol, Expr, Rel, Eq


dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([2, 128], "L_edge_index_"),
    ([128, 128], "L_self_modules_conv_modules_lin_parameters_weight_"),
    ([128], "L_self_modules_conv_parameters_bias_"),
    ([128], "L_self_modules_lin_parameters_bias_"),
    ([128, 128], "L_self_modules_lin_parameters_weight_"),
    ([128, 128], "L_x_"),
]
