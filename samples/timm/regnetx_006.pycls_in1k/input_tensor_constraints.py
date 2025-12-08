from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 528], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [48],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([48], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([48], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [48, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([48], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([48], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [48, 24, 3, 3],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([48], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([48], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [48, 48, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [48, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [96, 48, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [96, 24, 3, 3],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [96, 96, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [96, 48, 1, 1],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [96, 96, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [96, 24, 3, 3],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [96, 96, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [96, 96, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [96, 24, 3, 3],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([96], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([96], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [96, 96, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [240, 96, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [240, 24, 3, 3],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [240, 240, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [240, 96, 1, 1],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [240, 240, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [240, 24, 3, 3],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [240, 240, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [240, 240, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [240, 24, 3, 3],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [240, 240, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [240, 240, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [240, 24, 3, 3],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [240, 240, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [240, 240, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [240, 24, 3, 3],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [240, 240, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [528, 240, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [528, 24, 3, 3],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [528, 240, 1, 1],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [528, 24, 3, 3],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [528, 24, 3, 3],
        "L_self_modules_s4_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [528, 24, 3, 3],
        "L_self_modules_s4_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [528, 24, 3, 3],
        "L_self_modules_s4_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_weight_"),
    (
        [528, 24, 3, 3],
        "L_self_modules_s4_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b7_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_weight_"),
    (
        [528, 24, 3, 3],
        "L_self_modules_s4_modules_b7_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [528],
        "L_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([528], "L_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_bias_"),
    ([528], "L_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_weight_"),
    (
        [528, 528, 1, 1],
        "L_self_modules_s4_modules_b7_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_stem_modules_conv_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
