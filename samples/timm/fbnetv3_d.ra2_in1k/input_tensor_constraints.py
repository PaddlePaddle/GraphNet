from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_"),
    (
        [16],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([16], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_"),
    ([16], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_"),
    (
        [24, 1, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [16, 24, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([16], "L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_"),
    ([16], "L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_"),
    (
        [16],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([16], "L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_"),
    ([16], "L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_"),
    (
        [16, 1, 3, 3],
        "L_self_modules_blocks_modules_0_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [16, 16, 1, 1],
        "L_self_modules_blocks_modules_0_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_"),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_"),
    (
        [80, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [80, 16, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [24, 80, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_"),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_"),
    (
        [48, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [48, 24, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [24, 48, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_"),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_"),
    (
        [48, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [48, 24, 1, 1],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [24, 48, 1, 1],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_"),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_weight_"),
    (
        [48, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [48, 24, 1, 1],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [24, 48, 1, 1],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_weight_"),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_weight_"),
    (
        [48, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [48, 24, 1, 1],
        "L_self_modules_blocks_modules_1_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [24, 48, 1, 1],
        "L_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_5_modules_bn1_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_5_modules_bn1_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_5_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_5_modules_bn2_parameters_weight_"),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_1_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_1_modules_5_modules_bn3_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_1_modules_5_modules_bn3_parameters_weight_"),
    (
        [48, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [48, 24, 1, 1],
        "L_self_modules_blocks_modules_1_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [24, 48, 1, 1],
        "L_self_modules_blocks_modules_1_modules_5_modules_conv_pwl_parameters_weight_",
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
        [40],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_"),
    (
        [96, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [96, 24, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 96, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [96, 8, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [8, 96, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_"),
    (
        [120, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [120, 40, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [120, 16, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [16, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_"),
    (
        [120, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [120, 40, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [120, 16, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [16, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_"),
    (
        [120, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [120, 40, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [120, 16, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [16, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_weight_"),
    (
        [120, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [120, 40, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [120, 16, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [16, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [200],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [200],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([200], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([200], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [200],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [200],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([200], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([200], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
    (
        [72],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([72], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_"),
    ([72], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_"),
    (
        [200, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [200, 40, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [72, 200, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([216], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([216], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([216], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([216], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
    (
        [72],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([72], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_"),
    ([72], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_"),
    (
        [216, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [216, 72, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [72, 216, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([216], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([216], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([216], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([216], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
    (
        [72],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([72], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_"),
    ([72], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_"),
    (
        [216, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [216, 72, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [72, 216, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([216], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_"),
    ([216], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_"),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([216], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_"),
    ([216], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_"),
    (
        [72],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([72], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_"),
    ([72], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_"),
    (
        [216, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [216, 72, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [72, 216, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([216], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_"),
    ([216], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_"),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [216],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([216], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_"),
    ([216], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_"),
    (
        [72],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([72], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_"),
    ([72], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_"),
    (
        [216, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [216, 72, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [72, 216, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([360], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    ([360], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_"),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([360], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    ([360], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_"),
    (
        [360, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [360, 72, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 360, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [360, 24, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [24, 360, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_"),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_"),
    (
        [384, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [384, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [384, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_"),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_"),
    (
        [384, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [384, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [384, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_"),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_"),
    (
        [384, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [384, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [384, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_"),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_"),
    (
        [384, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [384, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [384, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_"),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_"),
    (
        [384, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [384, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [384, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_"),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_"),
    (
        [384, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [384, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [384, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 384, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_"),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_"),
    ([208], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_"),
    (
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 128, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 768, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1040], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1040], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_"),
    ([208], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_"),
    (
        [1040, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1040, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1040, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1040, 56, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [56, 1040, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1040], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_"),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1040], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_"),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_"),
    ([208], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_"),
    (
        [1040, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [1040, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1040, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1040, 56, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [56, 1040, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([1040], "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_"),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([1040], "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_"),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_"),
    ([208], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_"),
    (
        [1040, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [1040, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1040, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1040, 56, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [56, 1040, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([1040], "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_"),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([1040], "L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_"),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_"),
    ([208], "L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_"),
    (
        [1040, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [1040, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1040, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1040, 56, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [56, 1040, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([1040], "L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_"),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([1040], "L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_"),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_"),
    ([208], "L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_"),
    (
        [1040, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [1040, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1040, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1040],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1040, 56, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [56, 1040, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_"),
    (
        [1248, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [240, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 56, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [56, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([1440], "L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_"),
    (
        [1440],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [1440, 240, 1, 1],
        "L_self_modules_blocks_modules_6_modules_0_modules_conv_parameters_weight_",
    ),
    ([24], "L_self_modules_bn1_buffers_running_mean_"),
    ([24], "L_self_modules_bn1_buffers_running_var_"),
    ([24], "L_self_modules_bn1_parameters_bias_"),
    ([24], "L_self_modules_bn1_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1984], "L_self_modules_classifier_parameters_weight_"),
    ([1984, 1440, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([24, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
