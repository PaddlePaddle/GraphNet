from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 1125}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([], "L_self_modules_layernorm_eps"),
    ([192], "L_self_modules_layernorm_parameters_bias_"),
    ([192], "L_self_modules_layernorm_parameters_weight_"),
    ([192], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([192, 192], "L_self_modules_pooler_modules_dense_parameters_weight_"),
    ([S0, S1, 192], "dict_getitem_L_stack0_list_dict_keys_L_stack0_0_"),
    ([], "s14"),
]
