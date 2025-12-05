from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 640}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([16], "L_self_modules_model_modules_0_modules_bn_buffers_running_mean_"),
    ([16], "L_self_modules_model_modules_0_modules_bn_buffers_running_var_"),
    ([16], "L_self_modules_model_modules_0_modules_bn_parameters_bias_"),
    ([16], "L_self_modules_model_modules_0_modules_bn_parameters_weight_"),
    ([16, 3, 6, 6], "L_self_modules_model_modules_0_modules_conv_parameters_weight_"),
    (
        [128],
        "L_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_10_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_10_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_model_modules_10_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_10_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_10_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_model_modules_10_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_10_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_10_modules_cv3_modules_bn_buffers_running_var_",
    ),
    ([256], "L_self_modules_model_modules_10_modules_cv3_modules_bn_parameters_bias_"),
    (
        [256],
        "L_self_modules_model_modules_10_modules_cv3_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_model_modules_10_modules_cv3_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_10_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_10_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_10_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_10_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_model_modules_10_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_10_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_10_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_10_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_10_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_10_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_11_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_11_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_11_modules_cv1_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_11_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_model_modules_11_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_11_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_11_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([256], "L_self_modules_model_modules_11_modules_cv2_modules_bn_parameters_bias_"),
    (
        [256],
        "L_self_modules_model_modules_11_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 1, 1],
        "L_self_modules_model_modules_11_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([192], "L_self_modules_model_modules_12_modules_bn_buffers_running_mean_"),
    ([192], "L_self_modules_model_modules_12_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_12_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_12_modules_bn_parameters_weight_"),
    (
        [192, 256, 1, 1],
        "L_self_modules_model_modules_12_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_15_modules_cv1_modules_bn_parameters_weight_"),
    (
        [96, 384, 1, 1],
        "L_self_modules_model_modules_15_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_model_modules_15_modules_cv2_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_15_modules_cv2_modules_bn_parameters_weight_"),
    (
        [96, 384, 1, 1],
        "L_self_modules_model_modules_15_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_15_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_15_modules_cv3_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_15_modules_cv3_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_model_modules_15_modules_cv3_modules_bn_parameters_weight_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_model_modules_15_modules_cv3_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_15_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_model_modules_16_modules_bn_buffers_running_mean_"),
    ([128], "L_self_modules_model_modules_16_modules_bn_buffers_running_var_"),
    ([128], "L_self_modules_model_modules_16_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_16_modules_bn_parameters_weight_"),
    (
        [128, 192, 1, 1],
        "L_self_modules_model_modules_16_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_"),
    (
        [64, 256, 1, 1],
        "L_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_weight_"),
    (
        [64, 256, 1, 1],
        "L_self_modules_model_modules_19_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_19_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_19_modules_cv3_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_19_modules_cv3_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_19_modules_cv3_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_model_modules_19_modules_cv3_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_19_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_19_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([32], "L_self_modules_model_modules_1_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_model_modules_1_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_model_modules_1_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_1_modules_bn_parameters_weight_"),
    ([32, 16, 3, 3], "L_self_modules_model_modules_1_modules_conv_parameters_weight_"),
    ([64], "L_self_modules_model_modules_20_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_model_modules_20_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_model_modules_20_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_20_modules_bn_parameters_weight_"),
    (
        [64, 128, 1, 1],
        "L_self_modules_model_modules_20_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([32], "L_self_modules_model_modules_23_modules_cv1_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_23_modules_cv1_modules_bn_parameters_weight_"),
    (
        [32, 128, 1, 1],
        "L_self_modules_model_modules_23_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([32], "L_self_modules_model_modules_23_modules_cv2_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_23_modules_cv2_modules_bn_parameters_weight_"),
    (
        [32, 128, 1, 1],
        "L_self_modules_model_modules_23_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_23_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_23_modules_cv3_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_23_modules_cv3_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_23_modules_cv3_modules_bn_parameters_weight_"),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_23_modules_cv3_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 1, 1],
        "L_self_modules_model_modules_23_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_model_modules_23_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_model_modules_23_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([64], "L_self_modules_model_modules_24_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_model_modules_24_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_model_modules_24_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_24_modules_bn_parameters_weight_"),
    ([64, 64, 3, 3], "L_self_modules_model_modules_24_modules_conv_parameters_weight_"),
    (
        [64],
        "L_self_modules_model_modules_26_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_26_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_26_modules_cv1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_26_modules_cv1_modules_bn_parameters_weight_"),
    (
        [64, 128, 1, 1],
        "L_self_modules_model_modules_26_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_26_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_26_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_26_modules_cv2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_26_modules_cv2_modules_bn_parameters_weight_"),
    (
        [64, 128, 1, 1],
        "L_self_modules_model_modules_26_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_26_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_26_modules_cv3_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_26_modules_cv3_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_26_modules_cv3_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_model_modules_26_modules_cv3_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_26_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_26_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_26_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_26_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_26_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_26_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_26_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_26_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_26_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_26_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_model_modules_27_modules_bn_buffers_running_mean_"),
    ([128], "L_self_modules_model_modules_27_modules_bn_buffers_running_var_"),
    ([128], "L_self_modules_model_modules_27_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_27_modules_bn_parameters_weight_"),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_27_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_model_modules_29_modules_cv1_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_29_modules_cv1_modules_bn_parameters_weight_"),
    (
        [96, 256, 1, 1],
        "L_self_modules_model_modules_29_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_model_modules_29_modules_cv2_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_29_modules_cv2_modules_bn_parameters_weight_"),
    (
        [96, 256, 1, 1],
        "L_self_modules_model_modules_29_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_29_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_29_modules_cv3_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_29_modules_cv3_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_model_modules_29_modules_cv3_modules_bn_parameters_weight_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_model_modules_29_modules_cv3_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_model_modules_29_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_29_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_29_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([16], "L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_bias_"),
    ([16], "L_self_modules_model_modules_2_modules_cv1_modules_bn_parameters_weight_"),
    (
        [16, 32, 1, 1],
        "L_self_modules_model_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_model_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([16], "L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_"),
    ([16], "L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_"),
    (
        [16, 32, 1, 1],
        "L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_2_modules_cv3_modules_bn_buffers_running_var_",
    ),
    ([32], "L_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_2_modules_cv3_modules_bn_parameters_weight_"),
    (
        [32, 32, 1, 1],
        "L_self_modules_model_modules_2_modules_cv3_modules_conv_parameters_weight_",
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
        [16, 16, 1, 1],
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
    ([192], "L_self_modules_model_modules_30_modules_bn_buffers_running_mean_"),
    ([192], "L_self_modules_model_modules_30_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_30_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_30_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_30_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_32_modules_cv1_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_32_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 384, 1, 1],
        "L_self_modules_model_modules_32_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_32_modules_cv2_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_32_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 384, 1, 1],
        "L_self_modules_model_modules_32_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_32_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_32_modules_cv3_modules_bn_buffers_running_var_",
    ),
    ([256], "L_self_modules_model_modules_32_modules_cv3_modules_bn_parameters_bias_"),
    (
        [256],
        "L_self_modules_model_modules_32_modules_cv3_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_model_modules_32_modules_cv3_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_model_modules_32_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_32_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_model_modules_32_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_2_parameters_bias_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_33_modules_cv2_modules_0_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 128, 3, 3],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_2_parameters_bias_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_33_modules_cv2_modules_1_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 192, 3, 3],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_2_parameters_bias_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_33_modules_cv2_modules_2_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 256, 3, 3],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_2_parameters_bias_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_33_modules_cv2_modules_3_modules_2_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [80, 64, 3, 3],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [80, 80, 3, 3],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_2_parameters_bias_",
    ),
    (
        [80, 80, 1, 1],
        "L_self_modules_model_modules_33_modules_cv3_modules_0_modules_2_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [80, 128, 3, 3],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [80, 80, 3, 3],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_2_parameters_bias_",
    ),
    (
        [80, 80, 1, 1],
        "L_self_modules_model_modules_33_modules_cv3_modules_1_modules_2_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [80, 192, 3, 3],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [80, 80, 3, 3],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_2_parameters_bias_",
    ),
    (
        [80, 80, 1, 1],
        "L_self_modules_model_modules_33_modules_cv3_modules_2_modules_2_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [80, 256, 3, 3],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [80, 80, 3, 3],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_2_parameters_bias_",
    ),
    (
        [80, 80, 1, 1],
        "L_self_modules_model_modules_33_modules_cv3_modules_3_modules_2_parameters_weight_",
    ),
    (
        [1, 16, 1, 1],
        "L_self_modules_model_modules_33_modules_dfl_modules_conv_parameters_weight_",
    ),
    ([4], "L_self_modules_model_modules_33_stride"),
    ([64], "L_self_modules_model_modules_3_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_model_modules_3_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_model_modules_3_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_3_modules_bn_parameters_weight_"),
    ([64, 32, 3, 3], "L_self_modules_model_modules_3_modules_conv_parameters_weight_"),
    (
        [32],
        "L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([32], "L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_4_modules_cv1_modules_bn_parameters_weight_"),
    (
        [32, 64, 1, 1],
        "L_self_modules_model_modules_4_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_model_modules_4_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([32], "L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_4_modules_cv2_modules_bn_parameters_weight_"),
    (
        [32, 64, 1, 1],
        "L_self_modules_model_modules_4_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_cv3_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_4_modules_cv3_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_4_modules_cv3_modules_bn_parameters_weight_"),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_4_modules_cv3_modules_conv_parameters_weight_",
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
        [32, 32, 1, 1],
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
        [32, 32, 1, 1],
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
        [64],
        "L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_6_modules_cv1_modules_bn_parameters_weight_"),
    (
        [64, 128, 1, 1],
        "L_self_modules_model_modules_6_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_6_modules_cv2_modules_bn_parameters_weight_"),
    (
        [64, 128, 1, 1],
        "L_self_modules_model_modules_6_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_cv3_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_6_modules_cv3_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_6_modules_cv3_modules_bn_parameters_weight_"),
    (
        [128, 128, 1, 1],
        "L_self_modules_model_modules_6_modules_cv3_modules_conv_parameters_weight_",
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
        [64, 64, 1, 1],
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
        [64, 64, 1, 1],
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
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_6_modules_m_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([192], "L_self_modules_model_modules_7_modules_bn_buffers_running_mean_"),
    ([192], "L_self_modules_model_modules_7_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_7_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_7_modules_bn_parameters_weight_"),
    (
        [192, 128, 3, 3],
        "L_self_modules_model_modules_7_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_8_modules_cv1_modules_bn_parameters_weight_"),
    (
        [96, 192, 1, 1],
        "L_self_modules_model_modules_8_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_8_modules_cv2_modules_bn_parameters_weight_"),
    (
        [96, 192, 1, 1],
        "L_self_modules_model_modules_8_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_8_modules_cv3_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_8_modules_cv3_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_8_modules_cv3_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_8_modules_cv3_modules_bn_parameters_weight_"),
    (
        [192, 192, 1, 1],
        "L_self_modules_model_modules_8_modules_cv3_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_8_modules_m_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([256], "L_self_modules_model_modules_9_modules_bn_buffers_running_mean_"),
    ([256], "L_self_modules_model_modules_9_modules_bn_buffers_running_var_"),
    ([256], "L_self_modules_model_modules_9_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_9_modules_bn_parameters_weight_"),
    (
        [256, 192, 3, 3],
        "L_self_modules_model_modules_9_modules_conv_parameters_weight_",
    ),
    ([1, 3, S0, S0], "L_x_"),
]
