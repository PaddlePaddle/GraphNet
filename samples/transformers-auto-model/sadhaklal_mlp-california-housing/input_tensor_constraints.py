from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 3}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([50], "L_self_modules_fc1_parameters_bias_"),
    ([50, 8], "L_self_modules_fc1_parameters_weight_"),
    ([50], "L_self_modules_fc2_parameters_bias_"),
    ([50, 50], "L_self_modules_fc2_parameters_weight_"),
    ([50], "L_self_modules_fc3_parameters_bias_"),
    ([50, 50], "L_self_modules_fc3_parameters_weight_"),
    ([1], "L_self_modules_fc4_parameters_bias_"),
    ([1, 50], "L_self_modules_fc4_parameters_weight_"),
    ([S0, 8], "L_x_"),
    ([], "s77"),
]
