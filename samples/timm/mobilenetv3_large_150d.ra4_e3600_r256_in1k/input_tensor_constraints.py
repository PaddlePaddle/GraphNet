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
        [24, 1, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [24, 24, 1, 1],
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
        [96, 1, 3, 3],
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
        [120],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_"),
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
        [120, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [120, 40, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 120, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_"),
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
        [120, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [120, 40, 1, 1],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 120, 1, 1],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_"),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([64], "L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_"),
    ([64], "L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_"),
    (
        [120, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [120, 40, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [64, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [120, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_"),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([64], "L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_"),
    ([64], "L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_"),
    (
        [192, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [192, 64, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [64, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [192, 48, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_"),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([64], "L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_"),
    ([64], "L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_"),
    (
        [192, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [192, 64, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [64, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [192, 48, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_"),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([64], "L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_"),
    ([64], "L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_"),
    (
        [192, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [192, 64, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [64, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [192, 48, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [384],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_"),
    (
        [384, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [120, 384, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [304],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [304],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([304], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([304], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [304],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [304],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([304], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([304], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_"),
    (
        [304, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [304, 120, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [120, 304, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [304],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [304],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([304], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([304], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [304],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [304],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([304], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([304], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_"),
    (
        [304, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [304, 120, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [120, 304, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([280], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_"),
    ([280], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_"),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([280], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_"),
    ([280], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_"),
    (
        [280, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [280, 120, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [120, 280, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([280], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_"),
    ([280], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_"),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [280],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([280], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_"),
    ([280], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_"),
    (
        [280, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [280, 120, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [120, 280, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [720],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [720],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([720], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    ([720], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_"),
    (
        [720],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [720],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([720], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    ([720], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_"),
    (
        [168],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([168], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_"),
    ([168], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_"),
    (
        [720, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [720, 120, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [168, 720, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [720],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [720, 184, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [184],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [184, 720, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1008], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1008], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([168], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_"),
    ([168], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_"),
    (
        [1008, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1008, 168, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [168, 1008, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1008, 256, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [256, 1008, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1008], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_"),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1008], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_"),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([168], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_"),
    ([168], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_"),
    (
        [1008, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [1008, 168, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [168, 1008, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1008, 256, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [256, 1008, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([1008], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_"),
    (
        [1008],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([1008], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_"),
    (
        [1008],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_"),
    (
        [1008, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [1008, 168, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [240, 1008, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1008],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1008, 256, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [256, 1008, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1440], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1440], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_"),
    (
        [1440, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1440, 240, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [240, 1440, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1440, 360, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [360, 1440, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1440], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_"),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1440], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_"),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_"),
    (
        [1440, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [1440, 240, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [240, 1440, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1440, 360, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [360, 1440, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([1440], "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_"),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([1440], "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_"),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_"),
    (
        [1440, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [1440, 240, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [240, 1440, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1440, 360, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [360, 1440, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
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
    (
        [1440],
        "L_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1440],
        "L_self_modules_blocks_modules_6_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1440], "L_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_bias_"),
    (
        [1440],
        "L_self_modules_blocks_modules_6_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1440, 1440, 1, 1],
        "L_self_modules_blocks_modules_6_modules_1_modules_conv_parameters_weight_",
    ),
    ([24], "L_self_modules_bn1_buffers_running_mean_"),
    ([24], "L_self_modules_bn1_buffers_running_var_"),
    ([24], "L_self_modules_bn1_parameters_bias_"),
    ([24], "L_self_modules_bn1_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1280], "L_self_modules_classifier_parameters_weight_"),
    ([1280], "L_self_modules_conv_head_parameters_bias_"),
    ([1280, 1440, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([24, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
