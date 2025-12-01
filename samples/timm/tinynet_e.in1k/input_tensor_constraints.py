from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([32], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_"),
    ([32], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_"),
    (
        [8],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_",
    ),
    ([8], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_"),
    ([8], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_"),
    ([8], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_"),
    (
        [32, 1, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [8, 32, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [32, 8, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [8, 32, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_"),
    (
        [16],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([16], "L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_"),
    ([16], "L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_"),
    (
        [48, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [48, 8, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [16, 48, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [48, 2, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [2],
        "L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [2, 48, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([96], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_"),
    ([96], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_"),
    (
        [96],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([96], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_"),
    ([96], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_"),
    (
        [24],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_"),
    (
        [96, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [96, 16, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [24, 96, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [96, 4, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [4, 96, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([144], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([144], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [144],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([144], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([144], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_"),
    (
        [144, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [144, 24, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 144, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [144, 6, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [6],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [6, 144, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [240],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_"),
    (
        [240, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [240, 40, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 240, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [240, 10, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [10, 240, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_"),
    (
        [240],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_"),
    (
        [240, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [240, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [56, 240, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [240, 10, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [10, 240, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_"),
    (
        [336],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_"),
    (
        [336, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [336, 56, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [56, 336, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [336, 14, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [14],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [14, 336, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_"),
    (
        [336],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_"),
    (
        [96],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([96], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_"),
    ([96], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_"),
    (
        [336, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [336, 56, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [96, 336, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [336, 14, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [14],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [14, 336, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_"),
    (
        [576],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_"),
    (
        [96],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([96], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_"),
    ([96], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_"),
    (
        [576, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [576, 96, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [96, 576, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [576, 24, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [24, 576, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_"),
    (
        [576],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_"),
    (
        [576, 1, 3, 3],
        "L_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [576, 96, 1, 1],
        "L_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 576, 1, 1],
        "L_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [576, 24, 1, 1],
        "L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [24, 576, 1, 1],
        "L_self_modules_blocks_modules_6_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    ([32], "L_self_modules_bn1_buffers_running_mean_"),
    ([32], "L_self_modules_bn1_buffers_running_var_"),
    ([32], "L_self_modules_bn1_parameters_bias_"),
    ([32], "L_self_modules_bn1_parameters_weight_"),
    ([1280], "L_self_modules_bn2_buffers_running_mean_"),
    ([1280], "L_self_modules_bn2_buffers_running_var_"),
    ([1280], "L_self_modules_bn2_parameters_bias_"),
    ([1280], "L_self_modules_bn2_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1280], "L_self_modules_classifier_parameters_weight_"),
    ([1280, 160, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
