from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 158}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([256000, 2304], "L_self_modules_lm_head_parameters_weight_"),
    ([S0, S1, 2304], "L_stack0_last_hidden_state"),
]
