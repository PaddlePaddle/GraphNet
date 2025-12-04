dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([16], "L_self_modules_model_modules_0_modules_bn_buffers_running_mean_"),
    ([16], "L_self_modules_model_modules_0_modules_bn_buffers_running_var_"),
    ([16], "L_self_modules_model_modules_0_modules_bn_parameters_bias_"),
    ([16], "L_self_modules_model_modules_0_modules_bn_parameters_weight_"),
    ([16, 3, 3, 3], "L_self_modules_model_modules_0_modules_conv_parameters_weight_"),
    (
        [128],
        "L_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 384, 1, 1],
        "L_self_modules_model_modules_12_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_12_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_12_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_12_modules_cv2_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_12_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 192, 1, 1],
        "L_self_modules_model_modules_12_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_12_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_12_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_12_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_12_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_12_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_12_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_12_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_12_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_12_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_12_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_weight_"),
    (
        [64, 192, 1, 1],
        "L_self_modules_model_modules_15_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_15_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_15_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_15_modules_cv2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_15_modules_cv2_modules_bn_parameters_weight_"),
    (
        [64, 96, 1, 1],
        "L_self_modules_model_modules_15_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([64], "L_self_modules_model_modules_16_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_model_modules_16_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_model_modules_16_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_16_modules_bn_parameters_weight_"),
    ([64, 64, 3, 3], "L_self_modules_model_modules_16_modules_conv_parameters_weight_"),
    (
        [128],
        "L_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_18_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_18_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 192, 1, 1],
        "L_self_modules_model_modules_18_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_18_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_18_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_18_modules_cv2_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_18_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 192, 1, 1],
        "L_self_modules_model_modules_18_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_18_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_18_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_18_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_18_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_18_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_18_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_18_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_18_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_18_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_18_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_model_modules_19_modules_bn_buffers_running_mean_"),
    ([128], "L_self_modules_model_modules_19_modules_bn_buffers_running_var_"),
    ([128], "L_self_modules_model_modules_19_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_19_modules_bn_parameters_weight_"),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_19_modules_conv_parameters_weight_",
    ),
    ([32], "L_self_modules_model_modules_1_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_model_modules_1_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_model_modules_1_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_1_modules_bn_parameters_weight_"),
    ([32, 16, 3, 3], "L_self_modules_model_modules_1_modules_conv_parameters_weight_"),
    (
        [256],
        "L_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_21_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([256], "L_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_bias_"),
    (
        [256],
        "L_self_modules_model_modules_21_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 384, 1, 1],
        "L_self_modules_model_modules_21_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_21_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_21_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([256], "L_self_modules_model_modules_21_modules_cv2_modules_bn_parameters_bias_"),
    (
        [256],
        "L_self_modules_model_modules_21_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 384, 1, 1],
        "L_self_modules_model_modules_21_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_21_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_21_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_21_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_21_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_21_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_21_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_21_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_21_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_21_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_21_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_bias_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_22_modules_cv2_modules_0_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 128, 3, 3],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_bias_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_22_modules_cv2_modules_1_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 256, 3, 3],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_bias_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_22_modules_cv2_modules_2_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [15],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_bias_",
    ),
    (
        [15, 64, 1, 1],
        "L_self_modules_model_modules_22_modules_cv3_modules_0_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 128, 3, 3],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [15],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_bias_",
    ),
    (
        [15, 64, 1, 1],
        "L_self_modules_model_modules_22_modules_cv3_modules_1_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 256, 3, 3],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [15],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_bias_",
    ),
    (
        [15, 64, 1, 1],
        "L_self_modules_model_modules_22_modules_cv3_modules_2_modules_2_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [16, 64, 3, 3],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [16, 16, 3, 3],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_2_parameters_bias_",
    ),
    (
        [1, 16, 1, 1],
        "L_self_modules_model_modules_22_modules_cv4_modules_0_modules_2_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [16, 128, 3, 3],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [16, 16, 3, 3],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_2_parameters_bias_",
    ),
    (
        [1, 16, 1, 1],
        "L_self_modules_model_modules_22_modules_cv4_modules_1_modules_2_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [16, 256, 3, 3],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [16, 16, 3, 3],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_2_parameters_bias_",
    ),
    (
        [1, 16, 1, 1],
        "L_self_modules_model_modules_22_modules_cv4_modules_2_modules_2_parameters_weight_",
    ),
    (
        [1, 16, 1, 1],
        "L_self_modules_model_modules_22_modules_dfl_modules_conv_parameters_weight_",
    ),
    ([3], "L_self_modules_model_modules_22_stride"),
    (
        [32],
        "L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([32], "L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_"),
    (
        [32, 32, 1, 1],
        "L_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([32], "L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_"),
    (
        [32, 48, 1, 1],
        "L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [16, 16, 3, 3],
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
        [16],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [16, 16, 3, 3],
        "L_self_modules_model_modules_2_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([64], "L_self_modules_model_modules_3_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_model_modules_3_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_model_modules_3_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_3_modules_bn_parameters_weight_"),
    ([64, 32, 3, 3], "L_self_modules_model_modules_3_modules_conv_parameters_weight_"),
    (
        [64],
        "L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_"),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_"),
    (
        [64, 128, 1, 1],
        "L_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
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
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_model_modules_4_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_model_modules_5_modules_bn_buffers_running_mean_"),
    ([128], "L_self_modules_model_modules_5_modules_bn_buffers_running_var_"),
    ([128], "L_self_modules_model_modules_5_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_5_modules_bn_parameters_weight_"),
    ([128, 64, 3, 3], "L_self_modules_model_modules_5_modules_conv_parameters_weight_"),
    (
        [128],
        "L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_",
    ),
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
    ([128], "L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_"),
    (
        [128, 256, 1, 1],
        "L_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([256], "L_self_modules_model_modules_7_modules_bn_buffers_running_mean_"),
    ([256], "L_self_modules_model_modules_7_modules_bn_buffers_running_var_"),
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
    ([256], "L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_"),
    (
        [256, 384, 1, 1],
        "L_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_"),
    (
        [128, 256, 1, 1],
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
    ([256], "L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_"),
    (
        [256, 512, 1, 1],
        "L_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([1, 3, 640, 640], "L_x_"),
]
