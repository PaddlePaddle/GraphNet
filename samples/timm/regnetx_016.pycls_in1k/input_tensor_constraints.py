from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 912], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [72],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([72], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([72], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [72, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([72], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([72], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [72, 24, 3, 3],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([72], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([72], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [72, 72, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [72, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([72], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([72], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [72, 72, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([72], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([72], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [72, 24, 3, 3],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([72], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([72], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [72, 72, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [168, 72, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [168, 24, 3, 3],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [168, 168, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [168, 72, 1, 1],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [168, 168, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [168, 24, 3, 3],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [168, 168, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [168, 168, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [168, 24, 3, 3],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [168, 168, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [168, 168, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [168, 24, 3, 3],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [168, 168, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_"),
    (
        [408],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_"),
    (
        [408],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [408, 24, 3, 3],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_"),
    (
        [408],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [408, 168, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [408, 24, 3, 3],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [408, 168, 1, 1],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [408, 24, 3, 3],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [408, 24, 3, 3],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [408, 24, 3, 3],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [408, 24, 3, 3],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_"),
    (
        [408, 24, 3, 3],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_"),
    (
        [408, 24, 3, 3],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_"),
    (
        [408, 24, 3, 3],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_"),
    (
        [408, 24, 3, 3],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [408],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([408], "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_"),
    ([408], "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_"),
    (
        [408, 408, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([912], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([912], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [912, 408, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([912], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([912], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [912, 24, 3, 3],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([912], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([912], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [912, 912, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [912, 408, 1, 1],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([912], "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([912], "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [912, 912, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([912], "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([912], "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [912, 24, 3, 3],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [912],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([912], "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([912], "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [912, 912, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_stem_modules_conv_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
