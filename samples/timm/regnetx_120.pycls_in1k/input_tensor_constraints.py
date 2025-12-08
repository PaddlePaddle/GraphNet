from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 2240], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [224],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([224], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([224], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [224, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([224], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([224], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [224, 112, 3, 3],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([224], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([224], "L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [224, 224, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [224, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([224], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([224], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [224, 224, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([224], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([224], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [224, 112, 3, 3],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([224], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([224], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [224, 224, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_",
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
        [448, 224, 1, 1],
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
        [448, 112, 3, 3],
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
        [448, 224, 1, 1],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_",
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
        [448, 112, 3, 3],
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
        [448, 112, 3, 3],
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
        [448, 112, 3, 3],
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
    (
        [448],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [448, 448, 1, 1],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [448, 112, 3, 3],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([448], "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([448], "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [448, 448, 1, 1],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_conv_parameters_weight_",
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
        [896, 112, 3, 3],
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
    (
        [896],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_bias_"),
    (
        [896],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_bias_"),
    (
        [896],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [896, 112, 3, 3],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [896],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([896], "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_bias_"),
    (
        [896],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [896, 896, 1, 1],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_conv_parameters_weight_",
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
        [896, 112, 3, 3],
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
        [896, 112, 3, 3],
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
        [896, 112, 3, 3],
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
        [896, 112, 3, 3],
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
        [896, 112, 3, 3],
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
        [896, 112, 3, 3],
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
        [896, 112, 3, 3],
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
        [896, 112, 3, 3],
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
        [896, 112, 3, 3],
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
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([2240], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [2240, 896, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([2240], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [2240, 112, 3, 3],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([2240], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [2240, 2240, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [2240],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [2240, 896, 1, 1],
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
