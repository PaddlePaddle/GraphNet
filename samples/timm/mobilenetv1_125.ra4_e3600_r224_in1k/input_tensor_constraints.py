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
        [80],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_"),
    (
        [40, 1, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [80, 40, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_",
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
        [160],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_"),
    (
        [80, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [160, 80, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_",
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
        [160],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_"),
    (
        [160, 1, 3, 3],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [160, 160, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_",
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
        [320],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_"),
    (
        [160, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [320, 160, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_"),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_"),
    (
        [320, 1, 3, 3],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [320, 320, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
    (
        [320, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [640, 320, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
    (
        [640, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [640, 640, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
    (
        [640, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [640, 640, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_"),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_"),
    (
        [640, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [640, 640, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_"),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_"),
    (
        [640, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [640, 640, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_"),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_"),
    (
        [640, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [640, 640, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_"),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([1280], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_",
    ),
    (
        [640, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [1280, 640, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1280], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1280], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [1280, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1280, 1280, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_",
    ),
    ([40], "L_self_modules_bn1_buffers_running_mean_"),
    ([40], "L_self_modules_bn1_buffers_running_var_"),
    ([40], "L_self_modules_bn1_parameters_bias_"),
    ([40], "L_self_modules_bn1_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1280], "L_self_modules_classifier_parameters_weight_"),
    ([40, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
