from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_"),
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
        [128, 40, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [32, 128, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_"),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([32], "L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_"),
    ([32], "L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_"),
    (
        [128, 32, 3, 3],
        "L_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [32, 128, 1, 1],
        "L_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([256], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_"),
    ([256], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_"),
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
        [256, 32, 3, 3],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [40, 256, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_"),
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
        [320, 40, 3, 3],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [40, 320, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_"),
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
        [320, 40, 3, 3],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_",
    ),
    (
        [40, 320, 1, 1],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_"),
    (
        [320, 40, 3, 3],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [56, 320, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([448], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_"),
    (
        [448, 56, 3, 3],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [56, 448, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([448], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_"),
    (
        [448, 56, 3, 3],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_",
    ),
    (
        [56, 448, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([448], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_"),
    (
        [448, 56, 3, 3],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_",
    ),
    (
        [56, 448, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([448], "L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_"),
    (
        [448, 56, 3, 3],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_",
    ),
    (
        [56, 448, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([448], "L_self_modules_blocks_modules_2_modules_5_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_blocks_modules_2_modules_5_modules_bn1_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_2_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_2_modules_5_modules_bn2_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_2_modules_5_modules_bn2_parameters_weight_"),
    (
        [448, 56, 3, 3],
        "L_self_modules_blocks_modules_2_modules_5_modules_conv_exp_parameters_weight_",
    ),
    (
        [56, 448, 1, 1],
        "L_self_modules_blocks_modules_2_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([448], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [448],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([448], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([448], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([112], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_"),
    ([112], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_"),
    (
        [448, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [448, 56, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [112, 448, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([112], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_"),
    ([112], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_"),
    (
        [896, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [896, 112, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [112, 896, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([112], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_"),
    ([112], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_"),
    (
        [896, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [896, 112, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [112, 896, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_"),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_"),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([112], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_"),
    ([112], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_"),
    (
        [896, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [896, 112, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [112, 896, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_"),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_"),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([112], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_"),
    ([112], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_"),
    (
        [896, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [896, 112, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [112, 896, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_"),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_"),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([112], "L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_"),
    ([112], "L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_"),
    (
        [896, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [896, 112, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [112, 896, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_weight_"),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_weight_"),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_var_",
    ),
    ([112], "L_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_bias_"),
    ([112], "L_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_weight_"),
    (
        [896, 1, 5, 5],
        "L_self_modules_blocks_modules_3_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [896, 112, 1, 1],
        "L_self_modules_blocks_modules_3_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [112, 896, 1, 1],
        "L_self_modules_blocks_modules_3_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_"),
    (
        [896],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([896], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    ([896], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_"),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([176], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_"),
    ([176], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_"),
    (
        [896, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [896, 112, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [176, 896, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([176], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_"),
    ([176], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_"),
    (
        [1408, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1408, 176, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [176, 1408, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_",
    ),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([176], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_"),
    ([176], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_"),
    (
        [1408, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [1408, 176, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [176, 1408, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_",
    ),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([176], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_"),
    ([176], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_"),
    (
        [1408, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [1408, 176, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [176, 1408, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_",
    ),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([176], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_"),
    ([176], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_"),
    (
        [1408, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [1408, 176, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [176, 1408, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_",
    ),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [176],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([176], "L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_"),
    ([176], "L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_"),
    (
        [1408, 1, 5, 5],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [1408, 176, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [176, 1408, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [1408],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([1408], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_"),
    (
        [1408],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_",
    ),
    (
        [232],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [232],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([232], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_"),
    ([232], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_"),
    (
        [1408, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [1408, 176, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [232, 1408, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1856], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1856], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [232],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [232],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([232], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_"),
    ([232], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_"),
    (
        [1856, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1856, 232, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [232, 1856, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1856], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_"),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1856], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_"),
    (
        [1856],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_",
    ),
    (
        [232],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [232],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([232], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_"),
    ([232], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_"),
    (
        [1856, 1, 5, 5],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [1856, 232, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [232, 1856, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_",
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
    ([1536, 232, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([40, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
