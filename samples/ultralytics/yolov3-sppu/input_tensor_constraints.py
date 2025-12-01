dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([32], "L_self_modules_model_modules_0_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_model_modules_0_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_model_modules_0_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_model_modules_0_modules_bn_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_model_modules_0_modules_conv_parameters_weight_"),
    (
        [512],
        "L_self_modules_model_modules_10_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [512, 1024, 3, 3],
        "L_self_modules_model_modules_10_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [1024, 512, 3, 3],
        "L_self_modules_model_modules_10_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [512, 1024, 3, 3],
        "L_self_modules_model_modules_10_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [1024, 512, 3, 3],
        "L_self_modules_model_modules_10_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_2_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_2_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [512, 1024, 3, 3],
        "L_self_modules_model_modules_10_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_2_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_2_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [1024, 512, 3, 3],
        "L_self_modules_model_modules_10_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_3_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_3_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_3_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_10_modules_3_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [512, 1024, 3, 3],
        "L_self_modules_model_modules_10_modules_3_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_3_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_3_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_3_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_10_modules_3_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [1024, 512, 3, 3],
        "L_self_modules_model_modules_10_modules_3_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_11_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_11_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([512], "L_self_modules_model_modules_11_modules_cv1_modules_bn_parameters_bias_"),
    (
        [512],
        "L_self_modules_model_modules_11_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [512, 1024, 3, 3],
        "L_self_modules_model_modules_11_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_11_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_11_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([1024], "L_self_modules_model_modules_11_modules_cv2_modules_bn_parameters_bias_"),
    (
        [1024],
        "L_self_modules_model_modules_11_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [1024, 512, 3, 3],
        "L_self_modules_model_modules_11_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_12_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([512], "L_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_bias_"),
    (
        [512],
        "L_self_modules_model_modules_12_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [512, 1024, 1, 1],
        "L_self_modules_model_modules_12_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_12_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_12_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([512], "L_self_modules_model_modules_12_modules_cv2_modules_bn_parameters_bias_"),
    (
        [512],
        "L_self_modules_model_modules_12_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_model_modules_12_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([1024], "L_self_modules_model_modules_13_modules_bn_buffers_running_mean_"),
    ([1024], "L_self_modules_model_modules_13_modules_bn_buffers_running_var_"),
    ([1024], "L_self_modules_model_modules_13_modules_bn_parameters_bias_"),
    ([1024], "L_self_modules_model_modules_13_modules_bn_parameters_weight_"),
    (
        [1024, 512, 3, 3],
        "L_self_modules_model_modules_13_modules_conv_parameters_weight_",
    ),
    ([512], "L_self_modules_model_modules_14_modules_bn_buffers_running_mean_"),
    ([512], "L_self_modules_model_modules_14_modules_bn_buffers_running_var_"),
    ([512], "L_self_modules_model_modules_14_modules_bn_parameters_bias_"),
    ([512], "L_self_modules_model_modules_14_modules_bn_parameters_weight_"),
    (
        [512, 1024, 1, 1],
        "L_self_modules_model_modules_14_modules_conv_parameters_weight_",
    ),
    ([1024], "L_self_modules_model_modules_15_modules_bn_buffers_running_mean_"),
    ([1024], "L_self_modules_model_modules_15_modules_bn_buffers_running_var_"),
    ([1024], "L_self_modules_model_modules_15_modules_bn_parameters_bias_"),
    ([1024], "L_self_modules_model_modules_15_modules_bn_parameters_weight_"),
    (
        [1024, 512, 3, 3],
        "L_self_modules_model_modules_15_modules_conv_parameters_weight_",
    ),
    ([256], "L_self_modules_model_modules_16_modules_bn_buffers_running_mean_"),
    ([256], "L_self_modules_model_modules_16_modules_bn_buffers_running_var_"),
    ([256], "L_self_modules_model_modules_16_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_16_modules_bn_parameters_weight_"),
    (
        [256, 512, 1, 1],
        "L_self_modules_model_modules_16_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_19_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([256], "L_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_bias_"),
    (
        [256],
        "L_self_modules_model_modules_19_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 768, 3, 3],
        "L_self_modules_model_modules_19_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_19_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([512], "L_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_bias_"),
    (
        [512],
        "L_self_modules_model_modules_19_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_19_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([64], "L_self_modules_model_modules_1_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_model_modules_1_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_model_modules_1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_1_modules_bn_parameters_weight_"),
    ([64, 32, 3, 3], "L_self_modules_model_modules_1_modules_conv_parameters_weight_"),
    (
        [256],
        "L_self_modules_model_modules_20_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_20_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([256], "L_self_modules_model_modules_20_modules_cv1_modules_bn_parameters_bias_"),
    (
        [256],
        "L_self_modules_model_modules_20_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_model_modules_20_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_20_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_20_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([512], "L_self_modules_model_modules_20_modules_cv2_modules_bn_parameters_bias_"),
    (
        [512],
        "L_self_modules_model_modules_20_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_20_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([256], "L_self_modules_model_modules_21_modules_bn_buffers_running_mean_"),
    ([256], "L_self_modules_model_modules_21_modules_bn_buffers_running_var_"),
    ([256], "L_self_modules_model_modules_21_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_21_modules_bn_parameters_weight_"),
    (
        [256, 512, 1, 1],
        "L_self_modules_model_modules_21_modules_conv_parameters_weight_",
    ),
    ([512], "L_self_modules_model_modules_22_modules_bn_buffers_running_mean_"),
    ([512], "L_self_modules_model_modules_22_modules_bn_buffers_running_var_"),
    ([512], "L_self_modules_model_modules_22_modules_bn_parameters_bias_"),
    ([512], "L_self_modules_model_modules_22_modules_bn_parameters_weight_"),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_22_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_model_modules_23_modules_bn_buffers_running_mean_"),
    ([128], "L_self_modules_model_modules_23_modules_bn_buffers_running_var_"),
    ([128], "L_self_modules_model_modules_23_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_23_modules_bn_parameters_weight_"),
    (
        [128, 256, 1, 1],
        "L_self_modules_model_modules_23_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_26_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_26_modules_cv1_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_model_modules_26_modules_cv1_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_model_modules_26_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 384, 3, 3],
        "L_self_modules_model_modules_26_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_26_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_26_modules_cv2_modules_bn_buffers_running_var_",
    ),
    ([256], "L_self_modules_model_modules_26_modules_cv2_modules_bn_parameters_bias_"),
    (
        [256],
        "L_self_modules_model_modules_26_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_26_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_27_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_27_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_27_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_27_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_model_modules_27_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_27_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_27_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_27_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_27_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_27_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_27_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_27_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_27_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_27_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_model_modules_27_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_27_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_27_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_27_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_27_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_27_modules_1_modules_cv2_modules_conv_parameters_weight_",
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
        [64, 256, 3, 3],
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
        [64, 512, 3, 3],
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
        [64, 1024, 3, 3],
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
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_bias_",
    ),
    (
        [80, 256, 1, 1],
        "L_self_modules_model_modules_28_modules_cv3_modules_0_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_bias_",
    ),
    (
        [80, 256, 1, 1],
        "L_self_modules_model_modules_28_modules_cv3_modules_1_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [256, 1024, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_bias_",
    ),
    (
        [80, 256, 1, 1],
        "L_self_modules_model_modules_28_modules_cv3_modules_2_modules_2_parameters_weight_",
    ),
    (
        [1, 16, 1, 1],
        "L_self_modules_model_modules_28_modules_dfl_modules_conv_parameters_weight_",
    ),
    ([3], "L_self_modules_model_modules_28_stride"),
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
        [32, 64, 3, 3],
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
    ([64], "L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_model_modules_2_modules_cv2_modules_bn_parameters_weight_"),
    (
        [64, 32, 3, 3],
        "L_self_modules_model_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_model_modules_3_modules_bn_buffers_running_mean_"),
    ([128], "L_self_modules_model_modules_3_modules_bn_buffers_running_var_"),
    ([128], "L_self_modules_model_modules_3_modules_bn_parameters_bias_"),
    ([128], "L_self_modules_model_modules_3_modules_bn_parameters_weight_"),
    ([128, 64, 3, 3], "L_self_modules_model_modules_3_modules_conv_parameters_weight_"),
    (
        [64],
        "L_self_modules_model_modules_4_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_model_modules_4_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_model_modules_4_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [64, 128, 3, 3],
        "L_self_modules_model_modules_4_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_4_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_model_modules_4_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([256], "L_self_modules_model_modules_5_modules_bn_buffers_running_mean_"),
    ([256], "L_self_modules_model_modules_5_modules_bn_buffers_running_var_"),
    ([256], "L_self_modules_model_modules_5_modules_bn_parameters_bias_"),
    ([256], "L_self_modules_model_modules_5_modules_bn_parameters_weight_"),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_6_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_6_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_2_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_2_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_2_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_2_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_6_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_3_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_3_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_3_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_3_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_3_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_3_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_3_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_3_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_3_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_6_modules_3_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_4_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_4_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_4_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_4_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_4_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_4_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_4_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_4_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_4_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_6_modules_4_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_5_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_5_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_5_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_5_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_5_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_5_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_5_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_5_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_5_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_6_modules_5_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_6_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_6_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_6_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_6_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_6_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_6_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_6_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_6_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_6_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_6_modules_6_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_7_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_7_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_7_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_model_modules_6_modules_7_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_model_modules_6_modules_7_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_7_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_7_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_7_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_6_modules_7_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_model_modules_6_modules_7_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([512], "L_self_modules_model_modules_7_modules_bn_buffers_running_mean_"),
    ([512], "L_self_modules_model_modules_7_modules_bn_buffers_running_var_"),
    ([512], "L_self_modules_model_modules_7_modules_bn_parameters_bias_"),
    ([512], "L_self_modules_model_modules_7_modules_bn_parameters_weight_"),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_7_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_0_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_0_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_0_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_0_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_0_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_0_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_0_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_0_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_0_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_8_modules_0_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_1_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_1_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_1_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_1_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_1_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_1_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_1_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_1_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_1_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_8_modules_1_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_2_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_2_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_2_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_2_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_2_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_2_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_2_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_2_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_2_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_8_modules_2_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_3_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_3_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_3_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_3_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_3_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_3_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_3_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_3_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_3_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_8_modules_3_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_4_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_4_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_4_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_4_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_4_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_4_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_4_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_4_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_4_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_8_modules_4_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_5_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_5_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_5_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_5_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_5_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_5_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_5_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_5_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_5_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_8_modules_5_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_6_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_6_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_6_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_6_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_6_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_6_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_6_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_6_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_6_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_8_modules_6_modules_cv2_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_7_modules_cv1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_7_modules_cv1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_7_modules_cv1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_model_modules_8_modules_7_modules_cv1_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_model_modules_8_modules_7_modules_cv1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_7_modules_cv2_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_7_modules_cv2_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_7_modules_cv2_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_model_modules_8_modules_7_modules_cv2_modules_bn_parameters_weight_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_model_modules_8_modules_7_modules_cv2_modules_conv_parameters_weight_",
    ),
    ([1024], "L_self_modules_model_modules_9_modules_bn_buffers_running_mean_"),
    ([1024], "L_self_modules_model_modules_9_modules_bn_buffers_running_var_"),
    ([1024], "L_self_modules_model_modules_9_modules_bn_parameters_bias_"),
    ([1024], "L_self_modules_model_modules_9_modules_bn_parameters_weight_"),
    (
        [1024, 512, 3, 3],
        "L_self_modules_model_modules_9_modules_conv_parameters_weight_",
    ),
    ([1, 3, 640, 640], "L_x_"),
]
