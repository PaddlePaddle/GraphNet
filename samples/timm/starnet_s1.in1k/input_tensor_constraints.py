from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_parameters_bias_"),
    ([1000, 192], "L_self_modules_head_parameters_weight_"),
    ([192], "L_self_modules_norm_buffers_running_mean_"),
    ([192], "L_self_modules_norm_buffers_running_var_"),
    ([192], "L_self_modules_norm_parameters_bias_"),
    ([192], "L_self_modules_norm_parameters_weight_"),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_mean_",
    ),
    ([24], "L_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_var_"),
    ([24], "L_self_modules_stages_modules_0_modules_0_modules_bn_parameters_bias_"),
    ([24], "L_self_modules_stages_modules_0_modules_0_modules_bn_parameters_weight_"),
    ([24], "L_self_modules_stages_modules_0_modules_0_modules_conv_parameters_bias_"),
    (
        [24, 32, 3, 3],
        "L_self_modules_stages_modules_0_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [24, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [24, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [96, 24, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [96, 24, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [24, 96, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_2_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [24, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_2_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_2_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_2_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_2_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_2_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_2_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [24, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_2_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_2_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [96, 24, 1, 1],
        "L_self_modules_stages_modules_0_modules_2_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_2_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [96, 24, 1, 1],
        "L_self_modules_stages_modules_0_modules_2_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_2_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_2_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_2_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_2_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_0_modules_2_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [24, 96, 1, 1],
        "L_self_modules_stages_modules_0_modules_2_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_mean_",
    ),
    ([48], "L_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_var_"),
    ([48], "L_self_modules_stages_modules_1_modules_0_modules_bn_parameters_bias_"),
    ([48], "L_self_modules_stages_modules_1_modules_0_modules_bn_parameters_weight_"),
    ([48], "L_self_modules_stages_modules_1_modules_0_modules_conv_parameters_bias_"),
    (
        [48, 24, 3, 3],
        "L_self_modules_stages_modules_1_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [48, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [48, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [192, 48, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [192, 48, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [48, 192, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [48, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [48, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [192, 48, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [192, 48, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [48, 192, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_mean_",
    ),
    ([96], "L_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_stages_modules_2_modules_0_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_stages_modules_2_modules_0_modules_bn_parameters_weight_"),
    ([96], "L_self_modules_stages_modules_2_modules_0_modules_conv_parameters_bias_"),
    (
        [96, 48, 3, 3],
        "L_self_modules_stages_modules_2_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_7_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_7_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_7_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_7_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_7_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_7_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_7_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_7_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_7_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_7_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_7_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_7_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_7_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_7_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_7_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_7_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_7_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_7_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_8_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_8_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_8_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_8_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_8_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_8_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_8_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [96, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_8_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_8_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_8_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_8_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_2_modules_8_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_8_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_8_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_8_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_8_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_8_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_8_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_stages_modules_3_modules_0_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_stages_modules_3_modules_0_modules_bn_parameters_weight_"),
    ([192], "L_self_modules_stages_modules_3_modules_0_modules_conv_parameters_bias_"),
    (
        [192, 96, 3, 3],
        "L_self_modules_stages_modules_3_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_3_modules_dwconv2_modules_conv_parameters_bias_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_stages_modules_3_modules_3_modules_dwconv2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_3_modules_dwconv_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_3_modules_dwconv_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_3_modules_dwconv_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_3_modules_dwconv_modules_bn_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_3_modules_dwconv_modules_conv_parameters_bias_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_stages_modules_3_modules_3_modules_dwconv_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_3_modules_f1_modules_conv_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_stages_modules_3_modules_3_modules_f1_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_3_modules_f2_modules_conv_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_stages_modules_3_modules_3_modules_f2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_3_modules_g_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_3_modules_g_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_3_modules_g_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_3_modules_g_modules_bn_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_3_modules_3_modules_g_modules_conv_parameters_bias_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_stages_modules_3_modules_3_modules_g_modules_conv_parameters_weight_",
    ),
    ([32], "L_self_modules_stem_modules_0_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_stem_modules_0_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_stem_modules_0_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_stem_modules_0_modules_bn_parameters_weight_"),
    ([32], "L_self_modules_stem_modules_0_modules_conv_parameters_bias_"),
    ([32, 3, 3, 3], "L_self_modules_stem_modules_0_modules_conv_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
