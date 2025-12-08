from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([16], "L_self_modules_model_modules_0_modules_bn_buffers_running_mean_"),
    ([16], "L_self_modules_model_modules_0_modules_bn_buffers_running_var_"),
    ([], "L_self_modules_model_modules_0_modules_bn_eps"),
    ([], "L_self_modules_model_modules_0_modules_bn_momentum"),
    ([16], "L_self_modules_model_modules_0_modules_bn_parameters_bias_"),
    ([16], "L_self_modules_model_modules_0_modules_bn_parameters_weight_"),
    ([16, 3, 3, 3], "L_self_modules_model_modules_0_modules_conv_parameters_weight_"),
    (
        [1280],
        "L_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_mean_",
    ),
    (
        [1280],
        "L_self_modules_model_modules_10_modules_conv_modules_bn_buffers_running_var_",
    ),
    (
        [1280],
        "L_self_modules_model_modules_10_modules_conv_modules_bn_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_model_modules_10_modules_conv_modules_bn_parameters_weight_",
    ),
    (
        [1280, 256, 1, 1],
        "L_self_modules_model_modules_10_modules_conv_modules_conv_parameters_weight_",
    ),
    ([1000], "L_self_modules_model_modules_10_modules_linear_parameters_bias_"),
    ([1000, 1280], "L_self_modules_model_modules_10_modules_linear_parameters_weight_"),
    ([32], "L_self_modules_model_modules_1_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_model_modules_1_modules_bn_buffers_running_var_"),
    ([], "L_self_modules_model_modules_1_modules_bn_eps"),
    ([], "L_self_modules_model_modules_1_modules_bn_momentum"),
    ([32], "L_self_modules_model_modules_1_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_1_modules_bn_parameters_weight_"),
    ([32, 16, 3, 3], "L_self_modules_model_modules_1_modules_conv_parameters_weight_"),
    (
        [32],
        "L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_2_modules_cv1_modules_bn_eps"),
    ([], "L_self_modules_model_modules_2_modules_cv1_modules_bn_momentum"),
    ([32], "L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_"),
    (
        [32, 32, 1, 1],
        "L_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_2_modules_cv2_modules_bn_eps"),
    ([], "L_self_modules_model_modules_2_modules_cv2_modules_bn_momentum"),
    ([64], "L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_"),
    (
        [64, 48, 1, 1],
        "L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [8],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_momentum",
    ),
    (
        [8],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [8, 16, 3, 3],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_momentum",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [16, 8, 3, 3],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([64], "L_self_modules_model_modules_3_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_model_modules_3_modules_bn_buffers_running_var_"),
    ([], "L_self_modules_model_modules_3_modules_bn_eps"),
    ([], "L_self_modules_model_modules_3_modules_bn_momentum"),
    ([64], "L_self_modules_model_modules_3_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_3_modules_bn_parameters_weight_"),
    ([64, 64, 3, 3], "L_self_modules_model_modules_3_modules_conv_parameters_weight_"),
    (
        [64],
        "L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_4_modules_cv1_modules_bn_eps"),
    ([], "L_self_modules_model_modules_4_modules_cv1_modules_bn_momentum"),
    ([64], "L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_"),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_4_modules_cv2_modules_bn_eps"),
    ([], "L_self_modules_model_modules_4_modules_cv2_modules_bn_momentum"),
    ([128], "L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_"),
    (
        [128, 96, 1, 1],
        "L_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_momentum",
    ),
    (
        [16],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [16, 32, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_momentum",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [32, 16, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_model_modules_5_modules_bn_buffers_running_mean_"),
    ([128], "L_self_modules_model_modules_5_modules_bn_buffers_running_var_"),
    ([], "L_self_modules_model_modules_5_modules_bn_eps"),
    ([], "L_self_modules_model_modules_5_modules_bn_momentum"),
    ([128], "L_self_modules_model_modules_5_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_5_modules_bn_parameters_weight_"),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_6_modules_cv1_modules_bn_eps"),
    ([], "L_self_modules_model_modules_6_modules_cv1_modules_bn_momentum"),
    ([128], "L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_"),
    (
        [128, 128, 1, 1],
        "L_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_6_modules_cv2_modules_bn_eps"),
    ([], "L_self_modules_model_modules_6_modules_cv2_modules_bn_momentum"),
    ([128], "L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_"),
    (
        [128, 192, 1, 1],
        "L_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_momentum",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [32, 64, 1, 1],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_momentum",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [32, 64, 1, 1],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_momentum",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([256], "L_self_modules_model_modules_7_modules_bn_buffers_running_mean_"),
    ([256], "L_self_modules_model_modules_7_modules_bn_buffers_running_var_"),
    ([], "L_self_modules_model_modules_7_modules_bn_eps"),
    ([], "L_self_modules_model_modules_7_modules_bn_momentum"),
    ([256], "L_self_modules_model_modules_7_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_7_modules_bn_parameters_weight_"),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_7_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_8_modules_cv1_modules_bn_eps"),
    ([], "L_self_modules_model_modules_8_modules_cv1_modules_bn_momentum"),
    ([256], "L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_"),
    (
        [256, 256, 1, 1],
        "L_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_8_modules_cv2_modules_bn_eps"),
    ([], "L_self_modules_model_modules_8_modules_cv2_modules_bn_momentum"),
    ([256], "L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_"),
    (
        [256, 384, 1, 1],
        "L_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_momentum",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 128, 1, 1],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_momentum",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 128, 1, 1],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_momentum",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv3_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_momentum",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_momentum",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_momentum",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_momentum",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_9_modules_cv1_modules_bn_eps"),
    ([], "L_self_modules_model_modules_9_modules_cv1_modules_bn_momentum"),
    ([256], "L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_"),
    (
        [256, 256, 1, 1],
        "L_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_9_modules_cv2_modules_bn_eps"),
    ([], "L_self_modules_model_modules_9_modules_cv2_modules_bn_momentum"),
    ([256], "L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_"),
    (
        [256, 256, 1, 1],
        "L_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_bn_parameters_weight_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_pe_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_proj_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 1, 1],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_attn_modules_qkv_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 1, 1],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_model_modules_9_modules_m_modules_0_modules_ffn_modules_1_modules_conv_parameters_weight_",
    ),
    ([S0, 3, 640, 640], "L_x_"),
]
