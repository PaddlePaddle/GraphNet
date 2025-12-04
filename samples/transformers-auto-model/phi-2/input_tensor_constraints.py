from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 19}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1, 2560], "L_hidden_states_"),
    ([51200], "L_self_modules_linear_parameters_bias_"),
    ([51200, 2560], "L_self_modules_linear_parameters_weight_"),
    ([2560], "L_self_modules_ln_parameters_bias_"),
    ([2560], "L_self_modules_ln_parameters_weight_"),
]
