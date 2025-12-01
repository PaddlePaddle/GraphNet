from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_classifier_modules_1_parameters_bias_"),
    ([1000, 512, 1, 1], "L_self_modules_classifier_modules_1_parameters_weight_"),
    ([96], "L_self_modules_features_modules_0_parameters_bias_"),
    ([96, 3, 7, 7], "L_self_modules_features_modules_0_parameters_weight_"),
    ([256], "L_self_modules_features_modules_10_modules_expand1x1_parameters_bias_"),
    (
        [256, 64, 1, 1],
        "L_self_modules_features_modules_10_modules_expand1x1_parameters_weight_",
    ),
    ([256], "L_self_modules_features_modules_10_modules_expand3x3_parameters_bias_"),
    (
        [256, 64, 3, 3],
        "L_self_modules_features_modules_10_modules_expand3x3_parameters_weight_",
    ),
    ([64], "L_self_modules_features_modules_10_modules_squeeze_parameters_bias_"),
    (
        [64, 384, 1, 1],
        "L_self_modules_features_modules_10_modules_squeeze_parameters_weight_",
    ),
    ([256], "L_self_modules_features_modules_12_modules_expand1x1_parameters_bias_"),
    (
        [256, 64, 1, 1],
        "L_self_modules_features_modules_12_modules_expand1x1_parameters_weight_",
    ),
    ([256], "L_self_modules_features_modules_12_modules_expand3x3_parameters_bias_"),
    (
        [256, 64, 3, 3],
        "L_self_modules_features_modules_12_modules_expand3x3_parameters_weight_",
    ),
    ([64], "L_self_modules_features_modules_12_modules_squeeze_parameters_bias_"),
    (
        [64, 512, 1, 1],
        "L_self_modules_features_modules_12_modules_squeeze_parameters_weight_",
    ),
    ([64], "L_self_modules_features_modules_3_modules_expand1x1_parameters_bias_"),
    (
        [64, 16, 1, 1],
        "L_self_modules_features_modules_3_modules_expand1x1_parameters_weight_",
    ),
    ([64], "L_self_modules_features_modules_3_modules_expand3x3_parameters_bias_"),
    (
        [64, 16, 3, 3],
        "L_self_modules_features_modules_3_modules_expand3x3_parameters_weight_",
    ),
    ([16], "L_self_modules_features_modules_3_modules_squeeze_parameters_bias_"),
    (
        [16, 96, 1, 1],
        "L_self_modules_features_modules_3_modules_squeeze_parameters_weight_",
    ),
    ([64], "L_self_modules_features_modules_4_modules_expand1x1_parameters_bias_"),
    (
        [64, 16, 1, 1],
        "L_self_modules_features_modules_4_modules_expand1x1_parameters_weight_",
    ),
    ([64], "L_self_modules_features_modules_4_modules_expand3x3_parameters_bias_"),
    (
        [64, 16, 3, 3],
        "L_self_modules_features_modules_4_modules_expand3x3_parameters_weight_",
    ),
    ([16], "L_self_modules_features_modules_4_modules_squeeze_parameters_bias_"),
    (
        [16, 128, 1, 1],
        "L_self_modules_features_modules_4_modules_squeeze_parameters_weight_",
    ),
    ([128], "L_self_modules_features_modules_5_modules_expand1x1_parameters_bias_"),
    (
        [128, 32, 1, 1],
        "L_self_modules_features_modules_5_modules_expand1x1_parameters_weight_",
    ),
    ([128], "L_self_modules_features_modules_5_modules_expand3x3_parameters_bias_"),
    (
        [128, 32, 3, 3],
        "L_self_modules_features_modules_5_modules_expand3x3_parameters_weight_",
    ),
    ([32], "L_self_modules_features_modules_5_modules_squeeze_parameters_bias_"),
    (
        [32, 128, 1, 1],
        "L_self_modules_features_modules_5_modules_squeeze_parameters_weight_",
    ),
    ([128], "L_self_modules_features_modules_7_modules_expand1x1_parameters_bias_"),
    (
        [128, 32, 1, 1],
        "L_self_modules_features_modules_7_modules_expand1x1_parameters_weight_",
    ),
    ([128], "L_self_modules_features_modules_7_modules_expand3x3_parameters_bias_"),
    (
        [128, 32, 3, 3],
        "L_self_modules_features_modules_7_modules_expand3x3_parameters_weight_",
    ),
    ([32], "L_self_modules_features_modules_7_modules_squeeze_parameters_bias_"),
    (
        [32, 256, 1, 1],
        "L_self_modules_features_modules_7_modules_squeeze_parameters_weight_",
    ),
    ([192], "L_self_modules_features_modules_8_modules_expand1x1_parameters_bias_"),
    (
        [192, 48, 1, 1],
        "L_self_modules_features_modules_8_modules_expand1x1_parameters_weight_",
    ),
    ([192], "L_self_modules_features_modules_8_modules_expand3x3_parameters_bias_"),
    (
        [192, 48, 3, 3],
        "L_self_modules_features_modules_8_modules_expand3x3_parameters_weight_",
    ),
    ([48], "L_self_modules_features_modules_8_modules_squeeze_parameters_bias_"),
    (
        [48, 256, 1, 1],
        "L_self_modules_features_modules_8_modules_squeeze_parameters_weight_",
    ),
    ([192], "L_self_modules_features_modules_9_modules_expand1x1_parameters_bias_"),
    (
        [192, 48, 1, 1],
        "L_self_modules_features_modules_9_modules_expand1x1_parameters_weight_",
    ),
    ([192], "L_self_modules_features_modules_9_modules_expand3x3_parameters_bias_"),
    (
        [192, 48, 3, 3],
        "L_self_modules_features_modules_9_modules_expand3x3_parameters_weight_",
    ),
    ([48], "L_self_modules_features_modules_9_modules_squeeze_parameters_bias_"),
    (
        [48, 384, 1, 1],
        "L_self_modules_features_modules_9_modules_squeeze_parameters_weight_",
    ),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
