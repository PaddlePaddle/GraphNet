from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([64], "L_self_modules_features_modules_0_parameters_bias_"),
    ([64, 3, 3, 3], "L_self_modules_features_modules_0_parameters_weight_"),
    ([256], "L_self_modules_features_modules_10_parameters_bias_"),
    ([256, 128, 3, 3], "L_self_modules_features_modules_10_parameters_weight_"),
    ([256], "L_self_modules_features_modules_12_parameters_bias_"),
    ([256, 256, 3, 3], "L_self_modules_features_modules_12_parameters_weight_"),
    ([256], "L_self_modules_features_modules_14_parameters_bias_"),
    ([256, 256, 3, 3], "L_self_modules_features_modules_14_parameters_weight_"),
    ([512], "L_self_modules_features_modules_17_parameters_bias_"),
    ([512, 256, 3, 3], "L_self_modules_features_modules_17_parameters_weight_"),
    ([512], "L_self_modules_features_modules_19_parameters_bias_"),
    ([512, 512, 3, 3], "L_self_modules_features_modules_19_parameters_weight_"),
    ([512], "L_self_modules_features_modules_21_parameters_bias_"),
    ([512, 512, 3, 3], "L_self_modules_features_modules_21_parameters_weight_"),
    ([512], "L_self_modules_features_modules_24_parameters_bias_"),
    ([512, 512, 3, 3], "L_self_modules_features_modules_24_parameters_weight_"),
    ([512], "L_self_modules_features_modules_26_parameters_bias_"),
    ([512, 512, 3, 3], "L_self_modules_features_modules_26_parameters_weight_"),
    ([512], "L_self_modules_features_modules_28_parameters_bias_"),
    ([512, 512, 3, 3], "L_self_modules_features_modules_28_parameters_weight_"),
    ([64], "L_self_modules_features_modules_2_parameters_bias_"),
    ([64, 64, 3, 3], "L_self_modules_features_modules_2_parameters_weight_"),
    ([128], "L_self_modules_features_modules_5_parameters_bias_"),
    ([128, 64, 3, 3], "L_self_modules_features_modules_5_parameters_weight_"),
    ([128], "L_self_modules_features_modules_7_parameters_bias_"),
    ([128, 128, 3, 3], "L_self_modules_features_modules_7_parameters_weight_"),
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 4096], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([4096], "L_self_modules_pre_logits_modules_fc1_parameters_bias_"),
    ([4096, 512, 7, 7], "L_self_modules_pre_logits_modules_fc1_parameters_weight_"),
    ([4096], "L_self_modules_pre_logits_modules_fc2_parameters_bias_"),
    ([4096, 4096, 1, 1], "L_self_modules_pre_logits_modules_fc2_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
