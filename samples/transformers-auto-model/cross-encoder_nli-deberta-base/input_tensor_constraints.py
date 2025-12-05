from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 11}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([3072], "L_self_modules_intermediate_modules_dense_parameters_bias_"),
    ([3072, 768], "L_self_modules_intermediate_modules_dense_parameters_weight_"),
    ([768], "L_self_modules_output_modules_LayerNorm_parameters_bias_"),
    ([768], "L_self_modules_output_modules_LayerNorm_parameters_weight_"),
    ([], "L_self_modules_output_modules_LayerNorm_variance_epsilon"),
    ([768], "L_self_modules_output_modules_dense_parameters_bias_"),
    ([768, 3072], "L_self_modules_output_modules_dense_parameters_weight_"),
    ([], "L_self_modules_output_modules_dropout_p"),
    ([S0, S1, 768], "L_stack0_0_"),
    ([], "s40"),
]
