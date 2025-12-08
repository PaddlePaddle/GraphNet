from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")
S2 = Symbol("S2")

dynamic_dim_constraint_symbols = [S0, S1, S2]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 256, S2: 192}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 3, S1, S2], "L_inputs_"),
    (
        [64],
        "L_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_bias_",
    ),
    (
        [64, 3, 3, 3],
        "L_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_bias_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_bias_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_bias_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_bias_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_bias_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_bias_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_bias_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_bias_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_bias_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_bias_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_weight_",
    ),
    (
        [512, 256, 4, 4],
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
