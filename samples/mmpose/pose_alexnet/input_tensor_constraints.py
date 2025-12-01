from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")
S2 = Symbol("S2")

dynamic_dim_constraint_symbols = [S0, S1, S2]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 256, S2: 192}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 3, S1, S2], "L_inputs_"),
    ([64], "L_self_modules_backbone_modules_features_modules_0_parameters_bias_"),
    (
        [64, 3, 11, 11],
        "L_self_modules_backbone_modules_features_modules_0_parameters_weight_",
    ),
    ([256], "L_self_modules_backbone_modules_features_modules_10_parameters_bias_"),
    (
        [256, 256, 3, 3],
        "L_self_modules_backbone_modules_features_modules_10_parameters_weight_",
    ),
    ([192], "L_self_modules_backbone_modules_features_modules_3_parameters_bias_"),
    (
        [192, 64, 5, 5],
        "L_self_modules_backbone_modules_features_modules_3_parameters_weight_",
    ),
    ([384], "L_self_modules_backbone_modules_features_modules_6_parameters_bias_"),
    (
        [384, 192, 3, 3],
        "L_self_modules_backbone_modules_features_modules_6_parameters_weight_",
    ),
    ([256], "L_self_modules_backbone_modules_features_modules_8_parameters_bias_"),
    (
        [256, 384, 3, 3],
        "L_self_modules_backbone_modules_features_modules_8_parameters_weight_",
    ),
    (
        [256, 256, 4, 4],
        "L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_",
    ),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_"),
    (
        [256, 256, 4, 4],
        "L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_",
    ),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_"),
    (
        [256, 256, 4, 4],
        "L_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_",
    ),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_"),
    ([17], "L_self_modules_head_modules_final_layer_parameters_bias_"),
    ([17, 256, 1, 1], "L_self_modules_head_modules_final_layer_parameters_weight_"),
]
