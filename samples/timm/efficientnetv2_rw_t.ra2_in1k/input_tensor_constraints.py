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
        [24, 24, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_parameters_weight_",
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
        [24, 24, 3, 3],
        "L_self_modules_blocks_modules_0_modules_1_modules_conv_parameters_weight_",
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
        [40],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_"),
    (
        [96, 24, 3, 3],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [40, 96, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_"),
    (
        [160, 40, 3, 3],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [40, 160, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_"),
    (
        [160, 40, 3, 3],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_",
    ),
    (
        [40, 160, 1, 1],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_"),
    (
        [160, 40, 3, 3],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_exp_parameters_weight_",
    ),
    (
        [40, 160, 1, 1],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_",
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
        [48],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_"),
    (
        [160, 40, 3, 3],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [48, 160, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_",
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
        [48],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_"),
    (
        [192, 48, 3, 3],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [48, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_",
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
        [48],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_"),
    (
        [192, 48, 3, 3],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_",
    ),
    (
        [48, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_",
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
        [48],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_"),
    (
        [192, 48, 3, 3],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_",
    ),
    (
        [48, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
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
        [192, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [192, 48, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 192, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [192, 12, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [12, 192, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([416], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([416], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([416], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
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
        [416, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [416, 104, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 416, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [416, 26, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [26],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [26, 416, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([416], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([416], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([416], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
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
        [416, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [416, 104, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 416, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [416, 26, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [26],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [26, 416, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([416], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_"),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([416], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_"),
    ([416], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_"),
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
        [416, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [416, 104, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 416, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [416, 26, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [26],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [26, 416, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([416], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_"),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([416], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_"),
    ([416], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_"),
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
        [416, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [416, 104, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 416, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [416, 26, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [26],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [26, 416, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([416], "L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_"),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([416], "L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_"),
    ([416], "L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_"),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([104], "L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_"),
    ([104], "L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_"),
    (
        [416, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [416, 104, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 416, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [416, 26, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [26],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [26, 416, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([624], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    ([624], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_"),
    (
        [624],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([624], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    ([624], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_"),
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
        [624, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [624, 104, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 624, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [624, 26, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [26],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [26, 624, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_"),
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
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_"),
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
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_"),
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
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_"),
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
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_"),
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
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_"),
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
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_"),
    (
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_"),
    (
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_weight_",
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
        [1248],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_"),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_",
    ),
    (
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_"),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_",
    ),
    (
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_"),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_",
    ),
    (
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_"),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_",
    ),
    (
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    (
        [1248],
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
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_"),
    (
        [1248],
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
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_"),
    (
        [1248],
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
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_"),
    (
        [1248],
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
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_"),
    (
        [1248],
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
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
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
        [208],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_"),
    ([208], "L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_"),
    (
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_"),
    ([208], "L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_"),
    (
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_"),
    ([208], "L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_"),
    (
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_",
    ),
    ([1248], "L_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_"),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_",
    ),
    ([208], "L_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_"),
    ([208], "L_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_"),
    (
        [1248, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_",
    ),
    (
        [1248, 208, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_conv_pw_parameters_weight_",
    ),
    (
        [208, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1248, 52, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 1248, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    ([24], "L_self_modules_bn1_buffers_running_mean_"),
    ([24], "L_self_modules_bn1_buffers_running_var_"),
    ([24], "L_self_modules_bn1_parameters_bias_"),
    ([24], "L_self_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_bn2_buffers_running_mean_"),
    ([1024], "L_self_modules_bn2_buffers_running_var_"),
    ([1024], "L_self_modules_bn2_parameters_bias_"),
    ([1024], "L_self_modules_bn2_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1024], "L_self_modules_classifier_parameters_weight_"),
    ([1024, 208, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([24, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
