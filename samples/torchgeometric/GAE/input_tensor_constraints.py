from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1000}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 32], "L_args_0_"),
    ([2, 100], "L_args_1_"),
    ([16, 32], "L_self_modules_encoder_modules_lin_parameters_weight_"),
    ([16], "L_self_modules_encoder_parameters_bias_"),
]
