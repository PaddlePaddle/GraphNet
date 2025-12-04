from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 2}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([5], "L_self_modules_fc1_parameters_bias_"),
    ([5, 4], "L_self_modules_fc1_parameters_weight_"),
    ([3], "L_self_modules_fc2_parameters_bias_"),
    ([3, 5], "L_self_modules_fc2_parameters_weight_"),
    ([S0, 4], "L_x_"),
    ([], "s77"),
]
