from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [40],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_"),
    (
        [40],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([40], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_"),
    ([40], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_"),
    (
        [40, 1, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [40, 40, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_"),
    (
        [240],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_"),
    (
        [80, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [80, 1, 5, 5],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [80, 1, 7, 7],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [120, 20, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [120, 20, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [24, 120, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [24, 120, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([144], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_"),
    ([144], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_"),
    (
        [144],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([144], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_"),
    ([144], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_"),
    (
        [144, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [72, 24, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [72, 24, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [24, 72, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [24, 72, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [288],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([288], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_"),
    ([288], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_"),
    (
        [288],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [288],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([288], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_"),
    ([288], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_"),
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
        [72, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [72, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [72, 1, 7, 7],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [72, 1, 9, 9],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [288, 48, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [64, 288, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [288, 24, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [24, 288, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_"),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_"),
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
        [192, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [192, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [192, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [32, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [32, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [384, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 384, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_"),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_"),
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
        [192, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [192, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [192, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [32, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [32, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [384, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 384, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_"),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_"),
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
        [192, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [192, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [192, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [32, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [32, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [384, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 384, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_"),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_"),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([64], "L_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_bias_"),
    ([64], "L_self_modules_blocks_modules_2_modules_4_modules_bn3_parameters_weight_"),
    (
        [192, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [192, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [192, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [32, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [32, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [384, 32, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 384, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
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
        [128],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_"),
    (
        [128, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [128, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 384, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [384, 16, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [16, 384, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_"),
    (
        [192, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [192, 1, 9, 9],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [64, 384, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [64, 384, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_"),
    (
        [192, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [192, 1, 9, 9],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [64, 384, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [64, 384, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_"),
    (
        [192, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [192, 1, 9, 9],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [64, 384, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [64, 384, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_"),
    (
        [192, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [192, 1, 9, 9],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [64, 384, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [64, 384, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_"),
    (
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 64, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [64, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_"),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_"),
    (
        [144, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [144, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [144, 1, 7, 7],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [144, 1, 9, 9],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [288, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [288, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [96, 288, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [96, 288, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [576, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [96, 576, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_"),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_"),
    (
        [144, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [144, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [144, 1, 7, 7],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [144, 1, 9, 9],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [288, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [288, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [96, 288, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [96, 288, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [576, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [96, 576, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_"),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_"),
    (
        [144, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [144, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [144, 1, 7, 7],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [144, 1, 9, 9],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [288, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [288, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [96, 288, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [96, 288, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [576, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [96, 576, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_"),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([576], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_"),
    ([576], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_"),
    (
        [144, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [144, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [144, 1, 7, 7],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [144, 1, 9, 9],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [288, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [288, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [96, 288, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [96, 288, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [576, 96, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [96, 576, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_"),
    (
        [288, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [288, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [288, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [288, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [320, 1152, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 96, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [96, 1152, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1920], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1920], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_"),
    (
        [480, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [480, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [480, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [1920, 320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1920, 160, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [160, 1920, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1920], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_"),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1920], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_"),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_"),
    (
        [480, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [480, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [480, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [1920, 320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1920, 160, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [160, 1920, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([1920], "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_"),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([1920], "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_"),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_"),
    (
        [480, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [480, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [480, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [1920, 320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1920, 160, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [160, 1920, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([1920], "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_"),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([1920], "L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_"),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_"),
    (
        [480, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [480, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [480, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [480, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [1920, 320, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1920, 160, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [160, 1920, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    ([40], "L_self_modules_bn1_buffers_running_mean_"),
    ([40], "L_self_modules_bn1_buffers_running_var_"),
    ([40], "L_self_modules_bn1_parameters_bias_"),
    ([40], "L_self_modules_bn1_parameters_weight_"),
    ([1536], "L_self_modules_bn2_buffers_running_mean_"),
    ([1536], "L_self_modules_bn2_buffers_running_var_"),
    ([1536], "L_self_modules_bn2_parameters_bias_"),
    ([1536], "L_self_modules_bn2_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1536], "L_self_modules_classifier_parameters_weight_"),
    ([1536, 320, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([40, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
