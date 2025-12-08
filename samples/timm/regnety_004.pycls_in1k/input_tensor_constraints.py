from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 440], "L_self_modules_head_modules_fc_parameters_weight_"),
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
        [48, 8, 3, 3],
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
        [104],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [104, 48, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [104, 8, 3, 3],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [104, 104, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [104, 48, 1, 1],
        "L_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([12], "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [12, 104, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([104], "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [104, 12, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [104, 104, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [104, 8, 3, 3],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [104, 104, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([26], "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [26, 104, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([104], "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [104, 26, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [104, 104, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [104, 8, 3, 3],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [104, 104, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([26], "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [26, 104, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([104], "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [104, 26, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [208, 104, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [208, 8, 3, 3],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [208, 208, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [208, 104, 1, 1],
        "L_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([26], "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [26, 208, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([208], "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [208, 26, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [208, 208, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [208, 8, 3, 3],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [208, 208, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([52], "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [52, 208, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([208], "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [208, 52, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [208, 208, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [208, 8, 3, 3],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [208, 208, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([52], "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [52, 208, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([208], "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [208, 52, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [208, 208, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [208, 8, 3, 3],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [208, 208, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([52], "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [52, 208, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([208], "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [208, 52, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [208, 208, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [208, 8, 3, 3],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [208, 208, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([52], "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_"),
    (
        [52, 208, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_",
    ),
    ([208], "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_"),
    (
        [208, 52, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_"),
    (
        [208, 208, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_"),
    (
        [208, 8, 3, 3],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([208], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    ([208], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_"),
    (
        [208, 208, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([52], "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_"),
    (
        [52, 208, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_",
    ),
    ([208], "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_"),
    (
        [208, 52, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [440, 208, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [440, 8, 3, 3],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_"),
    (
        [440, 440, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_",
    ),
    (
        [440, 208, 1, 1],
        "L_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([52], "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [52, 440, 1, 1],
        "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([440], "L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [440, 52, 1, 1],
        "L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [440, 440, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [440, 8, 3, 3],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_"),
    (
        [440, 440, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([110], "L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [110, 440, 1, 1],
        "L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([440], "L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [440, 110, 1, 1],
        "L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [440, 440, 1, 1],
        "L_self_modules_s4_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [440, 8, 3, 3],
        "L_self_modules_s4_modules_b3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_weight_"),
    (
        [440, 440, 1, 1],
        "L_self_modules_s4_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([110], "L_self_modules_s4_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [110, 440, 1, 1],
        "L_self_modules_s4_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([440], "L_self_modules_s4_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [440, 110, 1, 1],
        "L_self_modules_s4_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [440, 440, 1, 1],
        "L_self_modules_s4_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [440, 8, 3, 3],
        "L_self_modules_s4_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [440, 440, 1, 1],
        "L_self_modules_s4_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([110], "L_self_modules_s4_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [110, 440, 1, 1],
        "L_self_modules_s4_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([440], "L_self_modules_s4_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [440, 110, 1, 1],
        "L_self_modules_s4_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [440, 440, 1, 1],
        "L_self_modules_s4_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [440, 8, 3, 3],
        "L_self_modules_s4_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [440, 440, 1, 1],
        "L_self_modules_s4_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([110], "L_self_modules_s4_modules_b5_modules_se_modules_fc1_parameters_bias_"),
    (
        [110, 440, 1, 1],
        "L_self_modules_s4_modules_b5_modules_se_modules_fc1_parameters_weight_",
    ),
    ([440], "L_self_modules_s4_modules_b5_modules_se_modules_fc2_parameters_bias_"),
    (
        [440, 110, 1, 1],
        "L_self_modules_s4_modules_b5_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_weight_"),
    (
        [440, 440, 1, 1],
        "L_self_modules_s4_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_weight_"),
    (
        [440, 8, 3, 3],
        "L_self_modules_s4_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [440],
        "L_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([440], "L_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    ([440], "L_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_weight_"),
    (
        [440, 440, 1, 1],
        "L_self_modules_s4_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([110], "L_self_modules_s4_modules_b6_modules_se_modules_fc1_parameters_bias_"),
    (
        [110, 440, 1, 1],
        "L_self_modules_s4_modules_b6_modules_se_modules_fc1_parameters_weight_",
    ),
    ([440], "L_self_modules_s4_modules_b6_modules_se_modules_fc2_parameters_bias_"),
    (
        [440, 110, 1, 1],
        "L_self_modules_s4_modules_b6_modules_se_modules_fc2_parameters_weight_",
    ),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_stem_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_stem_modules_bn_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_stem_modules_conv_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
