from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1536], "L_self_modules_classifier_modules_0_parameters_bias_"),
    ([1536], "L_self_modules_classifier_modules_0_parameters_weight_"),
    ([1000], "L_self_modules_classifier_modules_2_parameters_bias_"),
    ([1000, 1536], "L_self_modules_classifier_modules_2_parameters_weight_"),
    ([192], "L_self_modules_features_modules_0_modules_0_parameters_bias_"),
    ([192, 3, 4, 4], "L_self_modules_features_modules_0_modules_0_parameters_weight_"),
    ([192], "L_self_modules_features_modules_0_modules_1_parameters_bias_"),
    ([192], "L_self_modules_features_modules_0_modules_1_parameters_weight_"),
    (
        [192],
        "L_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_bias_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_features_modules_1_modules_0_modules_block_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_features_modules_1_modules_0_modules_block_modules_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_features_modules_1_modules_0_modules_block_modules_3_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_features_modules_1_modules_0_modules_block_modules_5_parameters_weight_",
    ),
    (
        [192, 1, 1],
        "L_self_modules_features_modules_1_modules_0_parameters_layer_scale_",
    ),
    (
        [192],
        "L_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_bias_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_features_modules_1_modules_1_modules_block_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_features_modules_1_modules_1_modules_block_modules_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_features_modules_1_modules_1_modules_block_modules_3_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_features_modules_1_modules_1_modules_block_modules_5_parameters_weight_",
    ),
    (
        [192, 1, 1],
        "L_self_modules_features_modules_1_modules_1_parameters_layer_scale_",
    ),
    (
        [192],
        "L_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_bias_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_features_modules_1_modules_2_modules_block_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_features_modules_1_modules_2_modules_block_modules_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_features_modules_1_modules_2_modules_block_modules_3_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_features_modules_1_modules_2_modules_block_modules_5_parameters_weight_",
    ),
    (
        [192, 1, 1],
        "L_self_modules_features_modules_1_modules_2_parameters_layer_scale_",
    ),
    ([192], "L_self_modules_features_modules_2_modules_0_parameters_bias_"),
    ([192], "L_self_modules_features_modules_2_modules_0_parameters_weight_"),
    ([384], "L_self_modules_features_modules_2_modules_1_parameters_bias_"),
    (
        [384, 192, 2, 2],
        "L_self_modules_features_modules_2_modules_1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_bias_",
    ),
    (
        [384, 1, 7, 7],
        "L_self_modules_features_modules_3_modules_0_modules_block_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_0_modules_block_modules_2_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_features_modules_3_modules_0_modules_block_modules_3_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_features_modules_3_modules_0_modules_block_modules_5_parameters_weight_",
    ),
    (
        [384, 1, 1],
        "L_self_modules_features_modules_3_modules_0_parameters_layer_scale_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_bias_",
    ),
    (
        [384, 1, 7, 7],
        "L_self_modules_features_modules_3_modules_1_modules_block_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_1_modules_block_modules_2_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_features_modules_3_modules_1_modules_block_modules_3_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_features_modules_3_modules_1_modules_block_modules_5_parameters_weight_",
    ),
    (
        [384, 1, 1],
        "L_self_modules_features_modules_3_modules_1_parameters_layer_scale_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_bias_",
    ),
    (
        [384, 1, 7, 7],
        "L_self_modules_features_modules_3_modules_2_modules_block_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_2_modules_block_modules_2_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_features_modules_3_modules_2_modules_block_modules_3_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_features_modules_3_modules_2_modules_block_modules_5_parameters_weight_",
    ),
    (
        [384, 1, 1],
        "L_self_modules_features_modules_3_modules_2_parameters_layer_scale_",
    ),
    ([384], "L_self_modules_features_modules_4_modules_0_parameters_bias_"),
    ([384], "L_self_modules_features_modules_4_modules_0_parameters_weight_"),
    ([768], "L_self_modules_features_modules_4_modules_1_parameters_bias_"),
    (
        [768, 384, 2, 2],
        "L_self_modules_features_modules_4_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_0_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_0_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_0_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_0_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_0_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_10_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_10_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_10_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_10_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_10_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_11_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_11_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_11_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_11_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_11_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_12_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_12_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_12_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_12_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_12_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_13_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_13_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_13_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_13_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_13_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_14_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_14_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_14_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_14_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_14_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_15_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_15_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_15_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_15_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_15_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_16_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_16_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_16_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_16_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_16_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_17_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_17_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_17_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_17_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_17_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_18_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_18_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_18_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_18_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_18_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_19_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_19_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_19_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_19_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_19_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_1_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_1_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_1_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_1_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_1_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_20_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_20_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_20_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_20_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_20_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_21_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_21_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_21_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_21_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_21_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_22_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_22_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_22_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_22_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_22_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_23_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_23_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_23_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_23_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_23_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_24_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_24_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_24_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_24_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_24_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_25_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_25_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_25_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_25_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_25_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_26_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_26_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_26_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_26_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_26_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_2_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_2_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_2_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_2_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_2_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_3_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_3_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_3_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_3_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_3_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_4_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_4_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_4_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_4_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_4_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_5_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_5_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_5_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_5_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_5_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_6_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_6_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_6_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_6_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_6_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_7_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_7_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_7_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_7_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_7_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_8_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_8_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_8_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_8_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_8_parameters_layer_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_features_modules_5_modules_9_modules_block_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_9_modules_block_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_5_modules_9_modules_block_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_5_modules_9_modules_block_modules_5_parameters_weight_",
    ),
    (
        [768, 1, 1],
        "L_self_modules_features_modules_5_modules_9_parameters_layer_scale_",
    ),
    ([768], "L_self_modules_features_modules_6_modules_0_parameters_bias_"),
    ([768], "L_self_modules_features_modules_6_modules_0_parameters_weight_"),
    ([1536], "L_self_modules_features_modules_6_modules_1_parameters_bias_"),
    (
        [1536, 768, 2, 2],
        "L_self_modules_features_modules_6_modules_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_bias_",
    ),
    (
        [1536, 1, 7, 7],
        "L_self_modules_features_modules_7_modules_0_modules_block_modules_0_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_0_modules_block_modules_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_features_modules_7_modules_0_modules_block_modules_3_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_features_modules_7_modules_0_modules_block_modules_5_parameters_weight_",
    ),
    (
        [1536, 1, 1],
        "L_self_modules_features_modules_7_modules_0_parameters_layer_scale_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_bias_",
    ),
    (
        [1536, 1, 7, 7],
        "L_self_modules_features_modules_7_modules_1_modules_block_modules_0_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_1_modules_block_modules_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_features_modules_7_modules_1_modules_block_modules_3_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_features_modules_7_modules_1_modules_block_modules_5_parameters_weight_",
    ),
    (
        [1536, 1, 1],
        "L_self_modules_features_modules_7_modules_1_parameters_layer_scale_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_bias_",
    ),
    (
        [1536, 1, 7, 7],
        "L_self_modules_features_modules_7_modules_2_modules_block_modules_0_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_2_modules_block_modules_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_features_modules_7_modules_2_modules_block_modules_3_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_features_modules_7_modules_2_modules_block_modules_5_parameters_weight_",
    ),
    (
        [1536, 1, 1],
        "L_self_modules_features_modules_7_modules_2_parameters_layer_scale_",
    ),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
