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
        [32],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([32], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_"),
    ([32], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_"),
    (
        [32, 1, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [32, 32, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_"),
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
        [64, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [64, 1, 5, 5],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [64, 1, 7, 7],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [96, 16, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [96, 16, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [20, 96, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [20, 96, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_1_parameters_weight_",
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
        [60, 20, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [60, 20, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [20, 60, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [20, 60, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_"),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_"),
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
        [60, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [60, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [60, 1, 7, 7],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [60, 1, 9, 9],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [240, 40, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [56, 240, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [240, 20, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [20, 240, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_"),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_"),
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
        [168, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [168, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [168, 28, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [168, 28, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [28, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [28, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [336, 28, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [28],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [28, 336, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_"),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_"),
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
        [168, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [168, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [168, 28, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [168, 28, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [28, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [28, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [336, 28, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [28],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [28, 336, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_"),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_"),
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
        [168, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [168, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [168, 28, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [168, 28, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [28, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [28, 168, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [336, 28, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [28],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [28, 336, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [336],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([336], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([336], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
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
        [112, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [112, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [112, 1, 7, 7],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [336, 56, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [104, 336, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [336, 14, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [14],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [14, 336, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([624], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([624], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([624], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([624], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
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
        [156, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [156, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [156, 1, 7, 7],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [156, 1, 9, 9],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [312, 52, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [312, 52, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [52, 312, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [52, 312, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [624, 26, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [26],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [26, 624, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([624], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([624], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([624], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([624], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
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
        [156, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [156, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [156, 1, 7, 7],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [156, 1, 9, 9],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [312, 52, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [312, 52, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [52, 312, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [52, 312, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [624, 26, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [26],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [26, 624, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([624], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_"),
    ([624], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_"),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([624], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_"),
    ([624], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_"),
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
        [156, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [156, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [156, 1, 7, 7],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [156, 1, 9, 9],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [312, 52, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [312, 52, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [52, 312, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [52, 312, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [624, 26, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [26],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [26, 624, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
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
        [624, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [624, 104, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 624, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [624],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [624, 52, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [52],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [52, 624, 1, 1],
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
        [120, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [120, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [120, 1, 7, 7],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [120, 1, 9, 9],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [240, 80, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [240, 80, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [80, 240, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [80, 240, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 80, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [80, 480, 1, 1],
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
        [120, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [120, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [120, 1, 7, 7],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [120, 1, 9, 9],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [240, 80, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [240, 80, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [80, 240, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [80, 240, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 80, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [80, 480, 1, 1],
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
        [120, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [120, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [120, 1, 7, 7],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [120, 1, 9, 9],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [240, 80, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [240, 80, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [80, 240, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [80, 240, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 80, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [80, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
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
        [240, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [240, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [240, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [240, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_3_parameters_weight_",
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
        [960, 80, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [80, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1584], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1584], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    (
        [1584],
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
        [396, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [396, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [396, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [396, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [1584, 264, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [132, 792, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [132, 792, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1584, 132, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [132],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [132, 1584, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1584], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_"),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1584], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_"),
    (
        [1584],
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
        [396, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [396, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [396, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [396, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [1584, 264, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [132, 792, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [132, 792, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1584, 132, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [132],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [132, 1584, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([1584], "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_"),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([1584], "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_"),
    (
        [1584],
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
        [396, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [396, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [396, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [396, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [1584, 264, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [132, 792, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [132, 792, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [1584],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1584, 132, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [132],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [132, 1584, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    ([32], "L_self_modules_bn1_buffers_running_mean_"),
    ([32], "L_self_modules_bn1_buffers_running_var_"),
    ([32], "L_self_modules_bn1_parameters_bias_"),
    ([32], "L_self_modules_bn1_parameters_weight_"),
    ([1536], "L_self_modules_bn2_buffers_running_mean_"),
    ([1536], "L_self_modules_bn2_buffers_running_var_"),
    ([1536], "L_self_modules_bn2_parameters_bias_"),
    ([1536], "L_self_modules_bn2_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1536], "L_self_modules_classifier_parameters_weight_"),
    ([1536, 264, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
