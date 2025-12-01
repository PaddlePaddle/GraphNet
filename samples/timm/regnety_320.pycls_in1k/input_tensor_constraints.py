from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 3712], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [232],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([232], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([232], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [232, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([232], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([232], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [232, 232, 3, 3],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([232], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([232], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [232, 232, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [232, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([8], "L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [8, 232, 1, 1],
        "L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([232], "L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [232, 8, 1, 1],
        "L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([232], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([232], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [232, 232, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([232], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([232], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [232, 232, 3, 3],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [232],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([232], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([232], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [232, 232, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([58], "L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [58, 232, 1, 1],
        "L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([232], "L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [232, 58, 1, 1],
        "L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [696, 232, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [696, 232, 3, 3],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [696, 696, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [696, 232, 1, 1],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([58], "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [58, 696, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([696], "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [696, 58, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [696, 696, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [696, 232, 3, 3],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [696, 696, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([174], "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [174, 696, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([696], "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [696, 174, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [696, 696, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [696, 232, 3, 3],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [696, 696, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([174], "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [174, 696, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([696], "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [696, 174, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [696, 696, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [696, 232, 3, 3],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [696, 696, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([174], "L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [174, 696, 1, 1],
        "L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([696], "L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [696, 174, 1, 1],
        "L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [696, 696, 1, 1],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [696, 232, 3, 3],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [696],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([696], "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([696], "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [696, 696, 1, 1],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([174], "L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_bias_"),
    (
        [174, 696, 1, 1],
        "L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_weight_",
    ),
    ([696], "L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_bias_"),
    (
        [696, 174, 1, 1],
        "L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 1392, 1, 1],
        "L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 348, 1, 1],
        "L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 1392, 1, 1],
        "L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 348, 1, 1],
        "L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 1392, 1, 1],
        "L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 348, 1, 1],
        "L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 696, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [1392, 696, 1, 1],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([174], "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [174, 1392, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 174, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 1392, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 348, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 1392, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 348, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 1392, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 348, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 1392, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 348, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 1392, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 348, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 1392, 1, 1],
        "L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 348, 1, 1],
        "L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 1392, 1, 1],
        "L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 348, 1, 1],
        "L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [1392, 232, 3, 3],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [1392],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([1392], "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_"),
    (
        [1392],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [1392, 1392, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 1392, 1, 1],
        "L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_",
    ),
    ([1392], "L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_"),
    (
        [1392, 348, 1, 1],
        "L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([3712], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [3712, 1392, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([3712], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [3712, 232, 3, 3],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([3712], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [3712, 3712, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [3712],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [3712, 1392, 1, 1],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([348], "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [348, 3712, 1, 1],
        "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([3712], "L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [3712, 348, 1, 1],
        "L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_stem_modules_conv_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
