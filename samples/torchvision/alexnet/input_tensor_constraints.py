from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([4096], "L_self_modules_classifier_modules_1_parameters_bias_"),
    ([4096, 9216], "L_self_modules_classifier_modules_1_parameters_weight_"),
    ([4096], "L_self_modules_classifier_modules_4_parameters_bias_"),
    ([4096, 4096], "L_self_modules_classifier_modules_4_parameters_weight_"),
    ([1000], "L_self_modules_classifier_modules_6_parameters_bias_"),
    ([1000, 4096], "L_self_modules_classifier_modules_6_parameters_weight_"),
    ([64], "L_self_modules_features_modules_0_parameters_bias_"),
    ([64, 3, 11, 11], "L_self_modules_features_modules_0_parameters_weight_"),
    ([256], "L_self_modules_features_modules_10_parameters_bias_"),
    ([256, 256, 3, 3], "L_self_modules_features_modules_10_parameters_weight_"),
    ([192], "L_self_modules_features_modules_3_parameters_bias_"),
    ([192, 64, 5, 5], "L_self_modules_features_modules_3_parameters_weight_"),
    ([384], "L_self_modules_features_modules_6_parameters_bias_"),
    ([384, 192, 3, 3], "L_self_modules_features_modules_6_parameters_weight_"),
    ([256], "L_self_modules_features_modules_8_parameters_bias_"),
    ([256, 384, 3, 3], "L_self_modules_features_modules_8_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
