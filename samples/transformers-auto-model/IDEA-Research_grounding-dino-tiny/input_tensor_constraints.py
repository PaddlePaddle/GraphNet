from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 900}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([], "L_self_activation_dropout"),
    ([], "L_self_dropout"),
    ([], "L_self_modules_encoder_attn_layer_norm_eps"),
    ([256], "L_self_modules_encoder_attn_layer_norm_parameters_bias_"),
    ([256], "L_self_modules_encoder_attn_layer_norm_parameters_weight_"),
    ([2048], "L_self_modules_fc1_parameters_bias_"),
    ([2048, 256], "L_self_modules_fc1_parameters_weight_"),
    ([256], "L_self_modules_fc2_parameters_bias_"),
    ([256, 2048], "L_self_modules_fc2_parameters_weight_"),
    ([], "L_self_modules_final_layer_norm_eps"),
    ([256], "L_self_modules_final_layer_norm_parameters_bias_"),
    ([256], "L_self_modules_final_layer_norm_parameters_weight_"),
    ([S0, S1, 256], "L_stack0_0_"),
    ([S0, S1, 256], "L_third_residual_"),
    ([], "s61"),
]
