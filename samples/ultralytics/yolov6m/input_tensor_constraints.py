from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([48], "L_self_modules_model_modules_0_modules_bn_buffers_running_mean_"),
    ([48], "L_self_modules_model_modules_0_modules_bn_buffers_running_var_"),
    ([48], "L_self_modules_model_modules_0_modules_bn_parameters_bias_"),
    ([48], "L_self_modules_model_modules_0_modules_bn_parameters_weight_"),
    ([48, 3, 3, 3], "L_self_modules_model_modules_0_modules_conv_parameters_weight_"),
    ([192], "L_self_modules_model_modules_10_modules_bn_buffers_running_mean_"),
    ([192], "L_self_modules_model_modules_10_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_10_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_10_modules_bn_parameters_weight_"),
    (
        [192, 576, 1, 1],
        "L_self_modules_model_modules_10_modules_conv_parameters_weight_",
    ),
    ([192], "L_self_modules_model_modules_11_parameters_bias_"),
    ([192, 192, 2, 2], "L_self_modules_model_modules_11_parameters_weight_"),
    ([192], "L_self_modules_model_modules_13_modules_bn_buffers_running_mean_"),
    ([192], "L_self_modules_model_modules_13_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_13_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_13_modules_bn_parameters_weight_"),
    (
        [192, 576, 3, 3],
        "L_self_modules_model_modules_13_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_0_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_14_modules_0_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_14_modules_0_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_14_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_1_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_14_modules_1_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_14_modules_1_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_14_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_2_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_14_modules_2_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_14_modules_2_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_14_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_3_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_14_modules_3_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_14_modules_3_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_14_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_4_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_14_modules_4_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_14_modules_4_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_14_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_14_modules_5_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_14_modules_5_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_14_modules_5_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_14_modules_5_modules_conv_parameters_weight_",
    ),
    ([96], "L_self_modules_model_modules_15_modules_bn_buffers_running_mean_"),
    ([96], "L_self_modules_model_modules_15_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_15_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_15_modules_bn_parameters_weight_"),
    (
        [96, 192, 1, 1],
        "L_self_modules_model_modules_15_modules_conv_parameters_weight_",
    ),
    ([96], "L_self_modules_model_modules_16_parameters_bias_"),
    ([96, 96, 2, 2], "L_self_modules_model_modules_16_parameters_weight_"),
    ([96], "L_self_modules_model_modules_18_modules_bn_buffers_running_mean_"),
    ([96], "L_self_modules_model_modules_18_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_18_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_18_modules_bn_parameters_weight_"),
    (
        [96, 288, 3, 3],
        "L_self_modules_model_modules_18_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_mean_",
    ),
    ([96], "L_self_modules_model_modules_19_modules_0_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_19_modules_0_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_19_modules_0_modules_bn_parameters_weight_"),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_19_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_mean_",
    ),
    ([96], "L_self_modules_model_modules_19_modules_1_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_19_modules_1_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_19_modules_1_modules_bn_parameters_weight_"),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_19_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_mean_",
    ),
    ([96], "L_self_modules_model_modules_19_modules_2_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_19_modules_2_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_19_modules_2_modules_bn_parameters_weight_"),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_19_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_mean_",
    ),
    ([96], "L_self_modules_model_modules_19_modules_3_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_19_modules_3_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_19_modules_3_modules_bn_parameters_weight_"),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_19_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_mean_",
    ),
    ([96], "L_self_modules_model_modules_19_modules_4_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_19_modules_4_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_19_modules_4_modules_bn_parameters_weight_"),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_19_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_mean_",
    ),
    ([96], "L_self_modules_model_modules_19_modules_5_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_19_modules_5_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_19_modules_5_modules_bn_parameters_weight_"),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_19_modules_5_modules_conv_parameters_weight_",
    ),
    ([96], "L_self_modules_model_modules_1_modules_bn_buffers_running_mean_"),
    ([96], "L_self_modules_model_modules_1_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_1_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_1_modules_bn_parameters_weight_"),
    ([96, 48, 3, 3], "L_self_modules_model_modules_1_modules_conv_parameters_weight_"),
    ([96], "L_self_modules_model_modules_20_modules_bn_buffers_running_mean_"),
    ([96], "L_self_modules_model_modules_20_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_20_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_20_modules_bn_parameters_weight_"),
    ([96, 96, 3, 3], "L_self_modules_model_modules_20_modules_conv_parameters_weight_"),
    ([192], "L_self_modules_model_modules_22_modules_bn_buffers_running_mean_"),
    ([192], "L_self_modules_model_modules_22_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_22_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_22_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_22_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_0_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_23_modules_0_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_23_modules_0_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_23_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_1_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_23_modules_1_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_23_modules_1_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_23_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_2_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_23_modules_2_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_23_modules_2_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_23_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_3_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_23_modules_3_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_23_modules_3_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_23_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_4_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_23_modules_4_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_23_modules_4_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_23_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_model_modules_23_modules_5_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_model_modules_23_modules_5_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_23_modules_5_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_23_modules_5_modules_conv_parameters_weight_",
    ),
    ([192], "L_self_modules_model_modules_24_modules_bn_buffers_running_mean_"),
    ([192], "L_self_modules_model_modules_24_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_24_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_24_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_24_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_model_modules_26_modules_bn_buffers_running_mean_"),
    ([384], "L_self_modules_model_modules_26_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_26_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_26_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_26_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_0_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_model_modules_27_modules_0_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_27_modules_0_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_27_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_1_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_model_modules_27_modules_1_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_27_modules_1_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_27_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_2_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_model_modules_27_modules_2_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_27_modules_2_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_27_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_3_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_model_modules_27_modules_3_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_27_modules_3_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_27_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_4_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_model_modules_27_modules_4_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_27_modules_4_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_27_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_model_modules_27_modules_5_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_model_modules_27_modules_5_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_27_modules_5_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_27_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 96, 3, 3],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_bias_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_28_modules_cv2_modules_0_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 192, 3, 3],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_bias_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_28_modules_cv2_modules_1_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [64, 384, 3, 3],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_bias_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_model_modules_28_modules_cv2_modules_2_modules_2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_bias_",
    ),
    (
        [80, 96, 1, 1],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [96, 192, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_bias_",
    ),
    (
        [80, 96, 1, 1],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [96, 384, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_bias_",
    ),
    (
        [80, 96, 1, 1],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_weight_",
    ),
    (
        [1, 16, 1, 1],
        "L_self_modules_model_modules_28_modules_dfl_modules_conv_parameters_weight_",
    ),
    ([3], "L_self_modules_model_modules_28_stride"),
    ([96], "L_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_mean_"),
    ([96], "L_self_modules_model_modules_2_modules_0_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_2_modules_0_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_2_modules_0_modules_bn_parameters_weight_"),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_2_modules_0_modules_conv_parameters_weight_",
    ),
    ([96], "L_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_mean_"),
    ([96], "L_self_modules_model_modules_2_modules_1_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_2_modules_1_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_2_modules_1_modules_bn_parameters_weight_"),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    ([96], "L_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_mean_"),
    ([96], "L_self_modules_model_modules_2_modules_2_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_2_modules_2_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_2_modules_2_modules_bn_parameters_weight_"),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_2_modules_2_modules_conv_parameters_weight_",
    ),
    ([96], "L_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_mean_"),
    ([96], "L_self_modules_model_modules_2_modules_3_modules_bn_buffers_running_var_"),
    ([96], "L_self_modules_model_modules_2_modules_3_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_model_modules_2_modules_3_modules_bn_parameters_weight_"),
    (
        [96, 96, 3, 3],
        "L_self_modules_model_modules_2_modules_3_modules_conv_parameters_weight_",
    ),
    ([192], "L_self_modules_model_modules_3_modules_bn_buffers_running_mean_"),
    ([192], "L_self_modules_model_modules_3_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_3_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_3_modules_bn_parameters_weight_"),
    ([192, 96, 3, 3], "L_self_modules_model_modules_3_modules_conv_parameters_weight_"),
    (
        [192],
        "L_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_mean_",
    ),
    ([192], "L_self_modules_model_modules_4_modules_0_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_4_modules_0_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_4_modules_0_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_4_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_mean_",
    ),
    ([192], "L_self_modules_model_modules_4_modules_1_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_4_modules_1_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_4_modules_1_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_4_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_mean_",
    ),
    ([192], "L_self_modules_model_modules_4_modules_2_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_4_modules_2_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_4_modules_2_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_4_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_mean_",
    ),
    ([192], "L_self_modules_model_modules_4_modules_3_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_4_modules_3_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_4_modules_3_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_4_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_mean_",
    ),
    ([192], "L_self_modules_model_modules_4_modules_4_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_4_modules_4_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_4_modules_4_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_4_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_mean_",
    ),
    ([192], "L_self_modules_model_modules_4_modules_5_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_4_modules_5_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_4_modules_5_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_4_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_mean_",
    ),
    ([192], "L_self_modules_model_modules_4_modules_6_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_4_modules_6_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_4_modules_6_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_4_modules_6_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_mean_",
    ),
    ([192], "L_self_modules_model_modules_4_modules_7_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_model_modules_4_modules_7_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_model_modules_4_modules_7_modules_bn_parameters_weight_"),
    (
        [192, 192, 3, 3],
        "L_self_modules_model_modules_4_modules_7_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_model_modules_5_modules_bn_buffers_running_mean_"),
    ([384], "L_self_modules_model_modules_5_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_5_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_5_modules_bn_parameters_weight_"),
    (
        [384, 192, 3, 3],
        "L_self_modules_model_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_mean_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_0_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_6_modules_0_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_0_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_10_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_10_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_10_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_10_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_11_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_11_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_11_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_11_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_mean_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_1_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_6_modules_1_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_1_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_mean_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_2_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_6_modules_2_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_2_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_mean_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_3_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_6_modules_3_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_3_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_mean_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_4_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_6_modules_4_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_4_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_mean_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_5_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_6_modules_5_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_5_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_mean_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_6_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_6_modules_6_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_6_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_6_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_mean_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_7_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_6_modules_7_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_7_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_7_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_mean_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_8_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_6_modules_8_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_8_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_8_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_mean_",
    ),
    ([384], "L_self_modules_model_modules_6_modules_9_modules_bn_buffers_running_var_"),
    ([384], "L_self_modules_model_modules_6_modules_9_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_model_modules_6_modules_9_modules_bn_parameters_weight_"),
    (
        [384, 384, 3, 3],
        "L_self_modules_model_modules_6_modules_9_modules_conv_parameters_weight_",
    ),
    ([576], "L_self_modules_model_modules_7_modules_bn_buffers_running_mean_"),
    ([576], "L_self_modules_model_modules_7_modules_bn_buffers_running_var_"),
    ([576], "L_self_modules_model_modules_7_modules_bn_parameters_bias_"),
    ([576], "L_self_modules_model_modules_7_modules_bn_parameters_weight_"),
    (
        [576, 384, 3, 3],
        "L_self_modules_model_modules_7_modules_conv_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_mean_",
    ),
    ([576], "L_self_modules_model_modules_8_modules_0_modules_bn_buffers_running_var_"),
    ([576], "L_self_modules_model_modules_8_modules_0_modules_bn_parameters_bias_"),
    ([576], "L_self_modules_model_modules_8_modules_0_modules_bn_parameters_weight_"),
    (
        [576, 576, 3, 3],
        "L_self_modules_model_modules_8_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_mean_",
    ),
    ([576], "L_self_modules_model_modules_8_modules_1_modules_bn_buffers_running_var_"),
    ([576], "L_self_modules_model_modules_8_modules_1_modules_bn_parameters_bias_"),
    ([576], "L_self_modules_model_modules_8_modules_1_modules_bn_parameters_weight_"),
    (
        [576, 576, 3, 3],
        "L_self_modules_model_modules_8_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_mean_",
    ),
    ([576], "L_self_modules_model_modules_8_modules_2_modules_bn_buffers_running_var_"),
    ([576], "L_self_modules_model_modules_8_modules_2_modules_bn_parameters_bias_"),
    ([576], "L_self_modules_model_modules_8_modules_2_modules_bn_parameters_weight_"),
    (
        [576, 576, 3, 3],
        "L_self_modules_model_modules_8_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_mean_",
    ),
    ([576], "L_self_modules_model_modules_8_modules_3_modules_bn_buffers_running_var_"),
    ([576], "L_self_modules_model_modules_8_modules_3_modules_bn_parameters_bias_"),
    ([576], "L_self_modules_model_modules_8_modules_3_modules_bn_parameters_weight_"),
    (
        [576, 576, 3, 3],
        "L_self_modules_model_modules_8_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [288],
        "L_self_modules_model_modules_9_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([288], "L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_bias_"),
    ([288], "L_self_modules_model_modules_9_modules_cv1_modules_bn_parameters_weight_"),
    (
        [288, 576, 1, 1],
        "L_self_modules_model_modules_9_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_model_modules_9_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([576], "L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_bias_"),
    ([576], "L_self_modules_model_modules_9_modules_cv2_modules_bn_parameters_weight_"),
    (
        [576, 1152, 1, 1],
        "L_self_modules_model_modules_9_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([S0, 3, 640, 640], "L_x_"),
]
