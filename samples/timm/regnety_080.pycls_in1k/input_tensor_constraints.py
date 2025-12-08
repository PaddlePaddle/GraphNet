from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 2016], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [168],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [168, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [168, 56, 3, 3],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [168, 168, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [168, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([8], "L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [8, 168, 1, 1],
        "L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([168], "L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [168, 8, 1, 1],
        "L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [168, 168, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [168, 56, 3, 3],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [168],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([168], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([168], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [168, 168, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([42], "L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [42, 168, 1, 1],
        "L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([168], "L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [168, 42, 1, 1],
        "L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [448, 168, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [448, 56, 3, 3],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [448, 448, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [448, 168, 1, 1],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([42], "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [42, 448, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([448], "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [448, 42, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [448, 448, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [448, 56, 3, 3],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [448, 448, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([112], "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [112, 448, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([448], "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [448, 112, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [448, 448, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [448, 56, 3, 3],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [448, 448, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([112], "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [112, 448, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([448], "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [448, 112, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [448, 448, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [448, 56, 3, 3],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [448, 448, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([112], "L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [112, 448, 1, 1],
        "L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([448], "L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [448, 112, 1, 1],
        "L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_"),
    (
        [896],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_"),
    (
        [896],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [896, 56, 3, 3],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_"),
    (
        [896],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([224], "L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_"),
    (
        [224, 896, 1, 1],
        "L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_",
    ),
    ([896], "L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_"),
    (
        [896, 224, 1, 1],
        "L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [896, 448, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [896, 56, 3, 3],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [896, 448, 1, 1],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([112], "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [112, 896, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([896], "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [896, 112, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [896, 56, 3, 3],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([224], "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [224, 896, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([896], "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [896, 224, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [896, 56, 3, 3],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([224], "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [224, 896, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([896], "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [896, 224, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [896, 56, 3, 3],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([224], "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [224, 896, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([896], "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [896, 224, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [896, 56, 3, 3],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([224], "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_"),
    (
        [224, 896, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_",
    ),
    ([896], "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_"),
    (
        [896, 224, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_"),
    (
        [896, 56, 3, 3],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([224], "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_"),
    (
        [224, 896, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_",
    ),
    ([896], "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_"),
    (
        [896, 224, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_"),
    (
        [896, 56, 3, 3],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([224], "L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_"),
    (
        [224, 896, 1, 1],
        "L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_",
    ),
    ([896], "L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_"),
    (
        [896, 224, 1, 1],
        "L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_"),
    (
        [896, 56, 3, 3],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([224], "L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_"),
    (
        [224, 896, 1, 1],
        "L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_",
    ),
    ([896], "L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_"),
    (
        [896, 224, 1, 1],
        "L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_"),
    (
        [896, 56, 3, 3],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_"),
    ([896], "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_"),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([224], "L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_"),
    (
        [224, 896, 1, 1],
        "L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_",
    ),
    ([896], "L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_"),
    (
        [896, 224, 1, 1],
        "L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([2016], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [2016, 896, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([2016], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [2016, 56, 3, 3],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([2016], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [2016, 2016, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [2016],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [2016, 896, 1, 1],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([224], "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [224, 2016, 1, 1],
        "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([2016], "L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [2016, 224, 1, 1],
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
