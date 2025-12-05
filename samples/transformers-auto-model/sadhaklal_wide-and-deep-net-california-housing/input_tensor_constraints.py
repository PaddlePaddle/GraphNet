from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 3}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([30], "L_self_modules_hidden1_parameters_bias_"),
    ([30, 8], "L_self_modules_hidden1_parameters_weight_"),
    ([30], "L_self_modules_hidden2_parameters_bias_"),
    ([30, 30], "L_self_modules_hidden2_parameters_weight_"),
    ([1], "L_self_modules_output_parameters_bias_"),
    ([1, 38], "L_self_modules_output_parameters_weight_"),
    ([S0, 8], "L_x_"),
    ([], "s77"),
]
