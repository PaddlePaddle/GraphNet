from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 3}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 6], "L_input_deep_"),
    ([S0, 5], "L_input_wide_"),
    ([1], "L_self_modules_aux_head_parameters_bias_"),
    ([1, 30], "L_self_modules_aux_head_parameters_weight_"),
    ([30], "L_self_modules_hidden1_parameters_bias_"),
    ([30, 6], "L_self_modules_hidden1_parameters_weight_"),
    ([30], "L_self_modules_hidden2_parameters_bias_"),
    ([30, 30], "L_self_modules_hidden2_parameters_weight_"),
    ([1], "L_self_modules_main_head_parameters_bias_"),
    ([1, 35], "L_self_modules_main_head_parameters_weight_"),
    ([], "s12"),
]
