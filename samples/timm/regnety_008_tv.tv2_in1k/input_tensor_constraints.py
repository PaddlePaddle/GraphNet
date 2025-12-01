from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 784], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [64],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [64, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [64, 16, 3, 3],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [64, 64, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [64, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([8], "L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [8, 64, 1, 1],
        "L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([64], "L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [64, 8, 1, 1],
        "L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([144], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([144], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [144, 64, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([144], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([144], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [144, 16, 3, 3],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([144], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([144], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [144, 144, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [144, 64, 1, 1],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([16], "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [16, 144, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([144], "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [144, 16, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([144], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([144], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [144, 144, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([144], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([144], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [144, 16, 3, 3],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([144], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([144], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [144, 144, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([36], "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [36, 144, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([144], "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [144, 36, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([144], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([144], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [144, 144, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([144], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([144], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [144, 16, 3, 3],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([144], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([144], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [144, 144, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([36], "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [36, 144, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([144], "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [144, 36, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [320, 144, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [320, 16, 3, 3],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [320, 144, 1, 1],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([36], "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [36, 320, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([320], "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [320, 36, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [320, 16, 3, 3],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([80], "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [80, 320, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([320], "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [320, 80, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [320, 16, 3, 3],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([80], "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [80, 320, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([320], "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [320, 80, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [320, 16, 3, 3],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([80], "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [80, 320, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([320], "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [320, 80, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [320, 16, 3, 3],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([80], "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_"),
    (
        [80, 320, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_",
    ),
    ([320], "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_"),
    (
        [320, 80, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_"),
    (
        [320, 16, 3, 3],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([80], "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_"),
    (
        [80, 320, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_",
    ),
    ([320], "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_"),
    (
        [320, 80, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_"),
    (
        [320, 16, 3, 3],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([80], "L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_"),
    (
        [80, 320, 1, 1],
        "L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_",
    ),
    ([320], "L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_"),
    (
        [320, 80, 1, 1],
        "L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_"),
    (
        [320, 16, 3, 3],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_"),
    (
        [320, 320, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([80], "L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_"),
    (
        [80, 320, 1, 1],
        "L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_",
    ),
    ([320], "L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_"),
    (
        [320, 80, 1, 1],
        "L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([784], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([784], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [784, 320, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([784], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([784], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [784, 16, 3, 3],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([784], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([784], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [784, 784, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [784, 320, 1, 1],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([80], "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [80, 784, 1, 1],
        "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([784], "L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [784, 80, 1, 1],
        "L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([784], "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([784], "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [784, 784, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([784], "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([784], "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [784, 16, 3, 3],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [784],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([784], "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([784], "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [784, 784, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([196], "L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [196, 784, 1, 1],
        "L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([784], "L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [784, 196, 1, 1],
        "L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_stem_modules_conv_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
