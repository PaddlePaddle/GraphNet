from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 1024], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_",
    ),
    (
        [256, 448, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_reduction_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_reduction_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_reduction_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_reduction_modules_bn_parameters_weight_",
    ),
    (
        [128, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_reduction_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_",
    ),
    (
        [512, 736, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [160, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [160, 160, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [160, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [160, 160, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_",
    ),
    (
        [160, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [160, 160, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_reduction_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_reduction_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_reduction_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_reduction_modules_bn_parameters_weight_",
    ),
    (
        [160, 256, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_reduction_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_",
    ),
    (
        [768, 1088, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [192, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [192, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_",
    ),
    (
        [192, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_reduction_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_reduction_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_reduction_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_reduction_modules_bn_parameters_weight_",
    ),
    (
        [192, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_reduction_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_fc_parameters_bias_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_fc_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1440, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_concat_modules_conv_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [224, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [224, 224, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [224, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [224, 224, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_buffers_running_var_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_bn_parameters_weight_",
    ),
    (
        [224, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [224, 224, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_mid_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_reduction_modules_bn_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_reduction_modules_bn_buffers_running_var_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_reduction_modules_bn_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_reduction_modules_bn_parameters_weight_",
    ),
    (
        [224, 768, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_reduction_modules_conv_parameters_weight_",
    ),
    ([64], "L_self_modules_stem_modules_0_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_stem_modules_0_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_stem_modules_0_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_stem_modules_0_modules_bn_parameters_weight_"),
    ([64, 3, 3, 3], "L_self_modules_stem_modules_0_modules_conv_parameters_weight_"),
    ([64], "L_self_modules_stem_modules_1_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_stem_modules_1_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_stem_modules_1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_stem_modules_1_modules_bn_parameters_weight_"),
    ([64, 1, 3, 3], "L_self_modules_stem_modules_1_modules_conv_dw_parameters_weight_"),
    (
        [64, 64, 1, 1],
        "L_self_modules_stem_modules_1_modules_conv_pw_parameters_weight_",
    ),
    ([64], "L_self_modules_stem_modules_2_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_stem_modules_2_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_stem_modules_2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_stem_modules_2_modules_bn_parameters_weight_"),
    ([64, 1, 3, 3], "L_self_modules_stem_modules_2_modules_conv_dw_parameters_weight_"),
    (
        [64, 64, 1, 1],
        "L_self_modules_stem_modules_2_modules_conv_pw_parameters_weight_",
    ),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
