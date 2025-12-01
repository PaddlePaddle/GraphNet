from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 888], "L_self_modules_head_modules_fc_parameters_weight_"),
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
    ([8], "L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [8, 48, 1, 1],
        "L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([48], "L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [48, 8, 1, 1],
        "L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([48], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([48], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [48, 48, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([48], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([48], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [48, 24, 3, 3],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([48], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([48], "L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [48, 48, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([12], "L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [12, 48, 1, 1],
        "L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([48], "L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [48, 12, 1, 1],
        "L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [120, 48, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [120, 24, 3, 3],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [120, 120, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [120, 48, 1, 1],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([12], "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [12, 120, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([120], "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [120, 12, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [120, 120, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [120, 24, 3, 3],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [120, 120, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([30], "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [30, 120, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([120], "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [120, 30, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [120, 120, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [120, 24, 3, 3],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [120, 120, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([30], "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [30, 120, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([120], "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [120, 30, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [120, 120, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [120, 24, 3, 3],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [120, 120, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([30], "L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [30, 120, 1, 1],
        "L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([120], "L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [120, 30, 1, 1],
        "L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [120, 120, 1, 1],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [120, 24, 3, 3],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [120, 120, 1, 1],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([30], "L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_bias_"),
    (
        [30, 120, 1, 1],
        "L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_weight_",
    ),
    ([120], "L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_bias_"),
    (
        [120, 30, 1, 1],
        "L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_weight_"),
    (
        [120, 120, 1, 1],
        "L_self_modules_s2_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_weight_"),
    (
        [120, 24, 3, 3],
        "L_self_modules_s2_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [120],
        "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([120], "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    ([120], "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_weight_"),
    (
        [120, 120, 1, 1],
        "L_self_modules_s2_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([30], "L_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_bias_"),
    (
        [30, 120, 1, 1],
        "L_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_weight_",
    ),
    ([120], "L_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_bias_"),
    (
        [120, 30, 1, 1],
        "L_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b14_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b14_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b14_modules_conv1_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b14_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b14_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b14_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b14_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b14_modules_conv2_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b14_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b14_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b14_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b14_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b14_modules_conv3_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b14_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b14_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b15_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b15_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b15_modules_conv1_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b15_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b15_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b15_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b15_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b15_modules_conv2_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b15_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b15_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b15_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b15_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b15_modules_conv3_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b15_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b15_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b15_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b15_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b15_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b15_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b16_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b16_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b16_modules_conv1_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b16_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b16_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b16_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b16_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b16_modules_conv2_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b16_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b16_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b16_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b16_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b16_modules_conv3_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b16_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b16_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b16_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b16_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b16_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b16_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b17_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b17_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b17_modules_conv1_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b17_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b17_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b17_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b17_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b17_modules_conv2_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b17_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b17_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b17_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b17_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b17_modules_conv3_modules_bn_parameters_bias_"),
    (
        [336],
        "L_self_modules_s3_modules_b17_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b17_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b17_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b17_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b17_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b17_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [336, 120, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [336, 120, 1, 1],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([30], "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [30, 336, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 30, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_"),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_"),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_"),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_"),
    (
        [336, 24, 3, 3],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [336],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([336], "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_"),
    ([336], "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_"),
    (
        [336, 336, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 336, 1, 1],
        "L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_",
    ),
    ([336], "L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_"),
    (
        [336, 84, 1, 1],
        "L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([888], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([888], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [888, 336, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([888], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([888], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [888, 24, 3, 3],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([888], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([888], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [888, 888, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [888, 336, 1, 1],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([84], "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [84, 888, 1, 1],
        "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([888], "L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [888, 84, 1, 1],
        "L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([888], "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([888], "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [888, 888, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([888], "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([888], "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [888, 24, 3, 3],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [888],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([888], "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([888], "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [888, 888, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([222], "L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [222, 888, 1, 1],
        "L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([888], "L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [888, 222, 1, 1],
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
