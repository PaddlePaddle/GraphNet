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
        [24],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_"),
    (
        [32, 1, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [24, 32, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_"),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_"),
    (
        [24, 1, 3, 3],
        "L_self_modules_blocks_modules_0_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [24, 24, 1, 1],
        "L_self_modules_blocks_modules_0_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_weight_"),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_weight_"),
    (
        [24, 1, 3, 3],
        "L_self_modules_blocks_modules_0_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [24, 24, 1, 1],
        "L_self_modules_blocks_modules_0_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([96], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_"),
    ([96], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_"),
    (
        [96],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([96], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_"),
    ([96], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_"),
    (
        [96, 1, 5, 5],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [96, 24, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 96, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_"),
    (
        [80, 1, 5, 5],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [80, 40, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 80, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_"),
    (
        [80, 1, 5, 5],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [80, 40, 1, 1],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 80, 1, 1],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_1_modules_3_modules_bn3_parameters_weight_"),
    (
        [80, 1, 5, 5],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [80, 40, 1, 1],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 80, 1, 1],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_1_modules_4_modules_bn3_parameters_weight_"),
    (
        [80, 1, 5, 5],
        "L_self_modules_blocks_modules_1_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [80, 40, 1, 1],
        "L_self_modules_blocks_modules_1_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 80, 1, 1],
        "L_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_"),
    (
        [160, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [160, 40, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [56, 160, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [160, 16, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [16, 160, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([168], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_"),
    ([168], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_"),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([168], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_"),
    ([168], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_"),
    (
        [168, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [168, 56, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [56, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [168, 16, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [16, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([168], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_"),
    ([168], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_"),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([168], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_"),
    ([168], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_"),
    (
        [168, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [168, 56, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [56, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [168, 16, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [16, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([168], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_"),
    ([168], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_"),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([168], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_"),
    ([168], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_"),
    (
        [168, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [168, 56, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [56, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [168, 16, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [16, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([168], "L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_"),
    ([168], "L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_"),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([168], "L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_"),
    ([168], "L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_weight_"),
    (
        [168, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [168, 56, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [56, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [168, 16, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [16, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([280], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([280], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([280], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([280], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([104], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_"),
    ([104], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_"),
    (
        [280, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [280, 56, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 280, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([312], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([312], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([312], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([312], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([104], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_"),
    ([104], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_"),
    (
        [312, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [312, 104, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 312, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([312], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([312], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([312], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([312], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([104], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_"),
    ([104], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_"),
    (
        [312, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [312, 104, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 312, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([312], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_"),
    ([312], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_"),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([312], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_"),
    ([312], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_"),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([104], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_"),
    ([104], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_"),
    (
        [312, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [312, 104, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 312, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([312], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_"),
    ([312], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_"),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [312],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([312], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_"),
    ([312], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_"),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([104], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_"),
    ([104], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_"),
    (
        [312, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [312, 104, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 312, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [520],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [520],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([520], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    ([520], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_"),
    (
        [520],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [520],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([520], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    ([520], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_"),
    (
        [520, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [520, 104, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 520, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [520],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [520, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 520, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_"),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_"),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [480, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_"),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_"),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [480, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_"),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_"),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [480, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_"),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_"),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [480, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_"),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_"),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [480, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_"),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_"),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [480, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_"),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_"),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_",
    ),
    (
        [480, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_"),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_"),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_",
    ),
    (
        [480, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_"),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([264], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_"),
    ([264], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [264, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [960, 40, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([264], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_"),
    ([264], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_"),
    (
        [1320, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1320, 264, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [264, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1320, 64, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [64, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([264], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_"),
    ([264], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_"),
    (
        [1320, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [1320, 264, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [264, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1320, 64, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [64, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([264], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_"),
    ([264], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_"),
    (
        [1320, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [1320, 264, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [264, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1320, 64, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [64, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([264], "L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_"),
    ([264], "L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_"),
    (
        [1320, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [1320, 264, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [264, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1320, 64, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [64, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([264], "L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_"),
    ([264], "L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_"),
    (
        [1320, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [1320, 264, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [264, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1320, 64, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [64, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_",
    ),
    ([1320], "L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_"),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_",
    ),
    (
        [264],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_",
    ),
    ([264], "L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_"),
    ([264], "L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_"),
    (
        [1320, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [1320, 264, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [264, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1320],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1320, 64, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [64, 1320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_",
    ),
    ([1584], "L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_"),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_",
    ),
    ([1584], "L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_"),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_",
    ),
    (
        [288],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_",
    ),
    ([288], "L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_"),
    ([288], "L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_"),
    (
        [1584, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_",
    ),
    (
        [1584, 264, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_pw_parameters_weight_",
    ),
    (
        [288, 1584, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1584, 64, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [64, 1584, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1728],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_",
    ),
    (
        [1728],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_",
    ),
    ([1728], "L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_"),
    (
        [1728],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_",
    ),
    (
        [1728],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_",
    ),
    (
        [1728],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_",
    ),
    ([1728], "L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_"),
    (
        [1728],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_",
    ),
    (
        [288],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_",
    ),
    ([288], "L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_"),
    ([288], "L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_"),
    (
        [1728, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_",
    ),
    (
        [1728, 288, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_pw_parameters_weight_",
    ),
    (
        [288, 1728, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1728],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1728, 72, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [72, 1728, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1728],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [1728],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([1728], "L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_"),
    (
        [1728],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [1728, 288, 1, 1],
        "L_self_modules_blocks_modules_6_modules_0_modules_conv_parameters_weight_",
    ),
    ([32], "L_self_modules_bn1_buffers_running_mean_"),
    ([32], "L_self_modules_bn1_buffers_running_var_"),
    ([32], "L_self_modules_bn1_parameters_bias_"),
    ([32], "L_self_modules_bn1_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1984], "L_self_modules_classifier_parameters_weight_"),
    ([1984, 1728, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
