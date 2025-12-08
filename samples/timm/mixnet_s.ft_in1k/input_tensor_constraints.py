from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [16],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([16], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_"),
    ([16], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_"),
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
        [16, 1, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [16, 16, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_",
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
        [96, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [48, 8, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [48, 8, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [12, 48, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [12, 48, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([72], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_"),
    ([72], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_"),
    (
        [72],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([72], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_"),
    ([72], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_"),
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
        [72, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [36, 12, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [36, 12, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [12, 36, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [12, 36, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([144], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_"),
    ([144], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_"),
    (
        [144],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([144], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_"),
    ([144], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_"),
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
        [48, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [48, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [48, 1, 7, 7],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [144, 24, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 144, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [144, 12, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [12, 144, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_"),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_"),
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
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [120, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [120, 20, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [120, 20, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [20, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [20, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [240, 20, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [20, 240, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_"),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_"),
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
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [120, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [120, 20, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [120, 20, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [20, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [20, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [240, 20, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [20, 240, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_"),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_"),
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
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [120, 1, 5, 5],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [120, 20, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [120, 20, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [20, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [20, 120, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [240, 20, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [20, 240, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [240],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_"),
    (
        [80, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [80, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [80, 1, 7, 7],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [240, 40, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 120, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [40, 120, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [240, 10, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [10, 240, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [480],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_"),
    (
        [240, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [240, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [480, 80, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 240, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [40, 240, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 20, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [20, 480, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [480],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_"),
    (
        [240, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [240, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [480, 80, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [40, 240, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [40, 240, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 20, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [20, 480, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_"),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([480], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    ([480], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_"),
    (
        [160, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [160, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [160, 1, 7, 7],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [240, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [240, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [60, 240, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [60, 240, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [480, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 480, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([360], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    ([360], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_"),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([360], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    ([360], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_"),
    (
        [90, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [90, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [90, 1, 7, 7],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [90, 1, 9, 9],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [180, 60, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [180, 60, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [60, 180, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [60, 180, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [360, 60, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [60],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [60, 360, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([360], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_"),
    ([360], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_"),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([360], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_"),
    ([360], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_"),
    (
        [120],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([120], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_"),
    ([120], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_"),
    (
        [90, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [90, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [90, 1, 7, 7],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [90, 1, 9, 9],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [180, 60, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_0_parameters_weight_",
    ),
    (
        [180, 60, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_modules_1_parameters_weight_",
    ),
    (
        [60, 180, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [60, 180, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [360, 60, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [60],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [60, 360, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [720],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [720],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([720], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_"),
    ([720], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_"),
    (
        [720],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [720],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([720], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_"),
    ([720], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_"),
    (
        [200],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [200],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([200], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_"),
    ([200], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_"),
    (
        [144, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [144, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [144, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [144, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [144, 1, 11, 11],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_modules_4_parameters_weight_",
    ),
    (
        [720, 120, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [200, 720, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [720],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [720, 60, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [60],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [60, 720, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1200], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1200], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [200],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [200],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([200], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_"),
    ([200], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_"),
    (
        [300, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [300, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [300, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [300, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [1200, 200, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [100, 600, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [100, 600, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1200, 100, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [100],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [100, 1200, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1200], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_"),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1200], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_"),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_",
    ),
    (
        [200],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [200],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([200], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_"),
    ([200], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_"),
    (
        [300, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_0_parameters_weight_",
    ),
    (
        [300, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_1_parameters_weight_",
    ),
    (
        [300, 1, 7, 7],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_2_parameters_weight_",
    ),
    (
        [300, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_modules_3_parameters_weight_",
    ),
    (
        [1200, 200, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [100, 600, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_0_parameters_weight_",
    ),
    (
        [100, 600, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_modules_1_parameters_weight_",
    ),
    (
        [1200],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1200, 100, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [100],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [100, 1200, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    ([16], "L_self_modules_bn1_buffers_running_mean_"),
    ([16], "L_self_modules_bn1_buffers_running_var_"),
    ([16], "L_self_modules_bn1_parameters_bias_"),
    ([16], "L_self_modules_bn1_parameters_weight_"),
    ([1536], "L_self_modules_bn2_buffers_running_mean_"),
    ([1536], "L_self_modules_bn2_buffers_running_var_"),
    ([1536], "L_self_modules_bn2_parameters_bias_"),
    ([1536], "L_self_modules_bn2_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1536], "L_self_modules_classifier_parameters_weight_"),
    ([1536, 200, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([16, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
