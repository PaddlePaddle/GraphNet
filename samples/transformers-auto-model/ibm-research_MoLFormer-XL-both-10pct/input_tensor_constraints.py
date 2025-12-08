from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 16}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_attention_mask_"),
    ([768], "L_self_modules_LayerNorm_parameters_bias_"),
    ([768], "L_self_modules_LayerNorm_parameters_weight_"),
    ([S0, S1, 768], "dict_getitem_L_stack0_list_dict_keys_L_stack0_0_"),
]
