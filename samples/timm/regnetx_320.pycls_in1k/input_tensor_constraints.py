from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 2520], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [336],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [336, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [336, 168, 3, 3],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [336, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [336, 168, 3, 3],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [672, 336, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [672, 168, 3, 3],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [672, 336, 1, 1],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [672, 168, 3, 3],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [672, 168, 3, 3],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [672, 168, 3, 3],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [672, 168, 3, 3],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_weight_"),
    (
        [672, 168, 3, 3],
        "L_self_modules_s2_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b7_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b7_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b7_modules_conv1_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b7_modules_conv1_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b7_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b7_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b7_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b7_modules_conv2_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b7_modules_conv2_modules_bn_parameters_weight_"),
    (
        [672, 168, 3, 3],
        "L_self_modules_s2_modules_b7_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b7_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [672],
        "L_self_modules_s2_modules_b7_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([672], "L_self_modules_s2_modules_b7_modules_conv3_modules_bn_parameters_bias_"),
    ([672], "L_self_modules_s2_modules_b7_modules_conv3_modules_bn_parameters_weight_"),
    (
        [672, 672, 1, 1],
        "L_self_modules_s2_modules_b7_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 672, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [1344, 672, 1, 1],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1344, 168, 3, 3],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1344],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1344], "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1344],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1344, 1344, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([2520], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [2520, 1344, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([2520], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [2520, 168, 3, 3],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([2520], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [2520, 2520, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [2520],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [2520, 1344, 1, 1],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_stem_modules_conv_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
