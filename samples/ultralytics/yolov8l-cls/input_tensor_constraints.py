from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 640}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([64], "L_self_modules_model_modules_0_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_model_modules_0_modules_bn_buffers_running_var_"),
    ([], "L_self_modules_model_modules_0_modules_bn_eps"),
    ([], "L_self_modules_model_modules_0_modules_bn_momentum"),
    ([64], "L_self_modules_model_modules_0_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_0_modules_bn_parameters_weight_"),
    ([64, 3, 3, 3], "L_self_modules_model_modules_0_modules_conv_parameters_weight_"),
    ([128], "L_self_modules_model_modules_1_modules_bn_buffers_running_mean_"),
    ([128], "L_self_modules_model_modules_1_modules_bn_buffers_running_var_"),
    ([], "L_self_modules_model_modules_1_modules_bn_eps"),
    ([], "L_self_modules_model_modules_1_modules_bn_momentum"),
    ([128], "L_self_modules_model_modules_1_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_1_modules_bn_parameters_weight_"),
    ([128, 64, 3, 3], "L_self_modules_model_modules_1_modules_conv_parameters_weight_"),
    (
        [128],
        "L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_2_modules_cv1_modules_bn_eps"),
    ([], "L_self_modules_model_modules_2_modules_cv1_modules_bn_momentum"),
    ([128], "L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_"),
    (
        [128, 128, 1, 1],
        "L_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_2_modules_cv2_modules_bn_eps"),
    ([], "L_self_modules_model_modules_2_modules_cv2_modules_bn_momentum"),
    ([128], "L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_"),
    (
        [128, 320, 1, 1],
        "L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
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
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
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
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_momentum",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_momentum",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_2_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv1_modules_bn_momentum",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv2_modules_bn_momentum",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_2_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([256], "L_self_modules_model_modules_3_modules_bn_buffers_running_mean_"),
    ([256], "L_self_modules_model_modules_3_modules_bn_buffers_running_var_"),
    ([], "L_self_modules_model_modules_3_modules_bn_eps"),
    ([], "L_self_modules_model_modules_3_modules_bn_momentum"),
    ([256], "L_self_modules_model_modules_3_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_3_modules_bn_parameters_weight_"),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_4_modules_cv1_modules_bn_eps"),
    ([], "L_self_modules_model_modules_4_modules_cv1_modules_bn_momentum"),
    ([256], "L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_"),
    (
        [256, 256, 1, 1],
        "L_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_4_modules_cv2_modules_bn_eps"),
    ([], "L_self_modules_model_modules_4_modules_cv2_modules_bn_momentum"),
    ([256], "L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_"),
    (
        [256, 1024, 1, 1],
        "L_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
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
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
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
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_momentum",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_momentum",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv1_modules_bn_momentum",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv2_modules_bn_momentum",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv1_modules_bn_momentum",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv2_modules_bn_momentum",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_3_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv1_modules_bn_momentum",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv2_modules_bn_momentum",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_4_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv1_modules_bn_momentum",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv2_modules_bn_momentum",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_5_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([512], "L_self_modules_model_modules_5_modules_bn_buffers_running_mean_"),
    ([512], "L_self_modules_model_modules_5_modules_bn_buffers_running_var_"),
    ([], "L_self_modules_model_modules_5_modules_bn_eps"),
    ([], "L_self_modules_model_modules_5_modules_bn_momentum"),
    ([512], "L_self_modules_model_modules_5_modules_bn_parameters_bias_"),
    ([512], "L_self_modules_model_modules_5_modules_bn_parameters_weight_"),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_6_modules_cv1_modules_bn_eps"),
    ([], "L_self_modules_model_modules_6_modules_cv1_modules_bn_momentum"),
    ([512], "L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_"),
    ([512], "L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_"),
    (
        [512, 512, 1, 1],
        "L_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_6_modules_cv2_modules_bn_eps"),
    ([], "L_self_modules_model_modules_6_modules_cv2_modules_bn_momentum"),
    ([512], "L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_"),
    ([512], "L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_"),
    (
        [512, 2048, 1, 1],
        "L_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
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
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
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
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_momentum",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_momentum",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_bn_momentum",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_bn_momentum",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv1_modules_bn_momentum",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv2_modules_bn_momentum",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_3_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv1_modules_bn_momentum",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv2_modules_bn_momentum",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_4_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv1_modules_bn_momentum",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv2_modules_bn_momentum",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_5_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([1024], "L_self_modules_model_modules_7_modules_bn_buffers_running_mean_"),
    ([1024], "L_self_modules_model_modules_7_modules_bn_buffers_running_var_"),
    ([], "L_self_modules_model_modules_7_modules_bn_eps"),
    ([], "L_self_modules_model_modules_7_modules_bn_momentum"),
    ([1024], "L_self_modules_model_modules_7_modules_bn_parameters_bias_"),
    ([1024], "L_self_modules_model_modules_7_modules_bn_parameters_weight_"),
    (
        [1024, 512, 3, 3],
        "L_self_modules_model_modules_7_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_8_modules_cv1_modules_bn_eps"),
    ([], "L_self_modules_model_modules_8_modules_cv1_modules_bn_momentum"),
    ([1024], "L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_"),
    (
        [1024],
        "L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([], "L_self_modules_model_modules_8_modules_cv2_modules_bn_eps"),
    ([], "L_self_modules_model_modules_8_modules_cv2_modules_bn_momentum"),
    ([1024], "L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_"),
    (
        [1024],
        "L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [1024, 2560, 1, 1],
        "L_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
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
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
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
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_momentum",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_momentum",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv1_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv1_modules_bn_momentum",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv2_modules_bn_eps",
    ),
    (
        [],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv2_modules_bn_momentum",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_mean_",
    ),
    (
        [1280],
        "L_self_modules_model_modules_9_modules_conv_modules_bn_buffers_running_var_",
    ),
    ([1280], "L_self_modules_model_modules_9_modules_conv_modules_bn_parameters_bias_"),
    (
        [1280],
        "L_self_modules_model_modules_9_modules_conv_modules_bn_parameters_weight_",
    ),
    (
        [1280, 1024, 1, 1],
        "L_self_modules_model_modules_9_modules_conv_modules_conv_parameters_weight_",
    ),
    ([1000], "L_self_modules_model_modules_9_modules_linear_parameters_bias_"),
    ([1000, 1280], "L_self_modules_model_modules_9_modules_linear_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
]
