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
        [192],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [192, 32, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [192, 8, 3, 3],
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
        [48, 192, 1, 1],
        "L_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([8], "L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [8, 192, 1, 1],
        "L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [192, 8, 1, 1],
        "L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [192, 48, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [192, 8, 3, 3],
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
        [48, 192, 1, 1],
        "L_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([12], "L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [12, 192, 1, 1],
        "L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [192, 12, 1, 1],
        "L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [416, 48, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [416, 8, 3, 3],
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
        [104, 416, 1, 1],
        "L_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([12], "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [12, 416, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([416], "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [416, 12, 1, 1],
        "L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [416, 104, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [416, 8, 3, 3],
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
        [104, 416, 1, 1],
        "L_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([26], "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [26, 416, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([416], "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [416, 26, 1, 1],
        "L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [416, 104, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [416, 8, 3, 3],
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
        [104, 416, 1, 1],
        "L_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([26], "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [26, 416, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([416], "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [416, 26, 1, 1],
        "L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [416, 104, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [416, 8, 3, 3],
        "L_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_"),
    (
        [104, 416, 1, 1],
        "L_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([26], "L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [26, 416, 1, 1],
        "L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([416], "L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [416, 26, 1, 1],
        "L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [416, 104, 1, 1],
        "L_self_modules_s2_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [416, 8, 3, 3],
        "L_self_modules_s2_modules_b5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_weight_"),
    (
        [104, 416, 1, 1],
        "L_self_modules_s2_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([26], "L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_bias_"),
    (
        [26, 416, 1, 1],
        "L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_weight_",
    ),
    ([416], "L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_bias_"),
    (
        [416, 26, 1, 1],
        "L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_weight_"),
    (
        [416, 104, 1, 1],
        "L_self_modules_s2_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [416],
        "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([416], "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    ([416], "L_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_weight_"),
    (
        [416, 8, 3, 3],
        "L_self_modules_s2_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([104], "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    ([104], "L_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_weight_"),
    (
        [104, 416, 1, 1],
        "L_self_modules_s2_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([26], "L_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_bias_"),
    (
        [26, 416, 1, 1],
        "L_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_weight_",
    ),
    ([416], "L_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_bias_"),
    (
        [416, 26, 1, 1],
        "L_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_"),
    (
        [960],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_"),
    (
        [960],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [960, 8, 3, 3],
        "L_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_"),
    (
        [240],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_bias_"),
    (
        [960],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b11_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_bias_"),
    (
        [960],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [960, 8, 3, 3],
        "L_self_modules_s3_modules_b11_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_bias_"),
    (
        [240],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b11_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_bias_"),
    (
        [960],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b12_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_bias_"),
    (
        [960],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [960, 8, 3, 3],
        "L_self_modules_s3_modules_b12_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_bias_"),
    (
        [240],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b12_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_parameters_bias_"),
    (
        [960],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b13_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_parameters_bias_"),
    (
        [960],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [960, 8, 3, 3],
        "L_self_modules_s3_modules_b13_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_parameters_bias_"),
    (
        [240],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b13_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b14_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b14_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b14_modules_conv1_modules_bn_parameters_bias_"),
    (
        [960],
        "L_self_modules_s3_modules_b14_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b14_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b14_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b14_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b14_modules_conv2_modules_bn_parameters_bias_"),
    (
        [960],
        "L_self_modules_s3_modules_b14_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [960, 8, 3, 3],
        "L_self_modules_s3_modules_b14_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b14_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b14_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b14_modules_conv3_modules_bn_parameters_bias_"),
    (
        [240],
        "L_self_modules_s3_modules_b14_modules_conv3_modules_bn_parameters_weight_",
    ),
    (
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b14_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_"),
    (
        [960, 104, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_"),
    (
        [960, 8, 3, 3],
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
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([26], "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [26, 960, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 26, 1, 1],
        "L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_"),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_"),
    (
        [960, 8, 3, 3],
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
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_"),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_"),
    (
        [960, 8, 3, 3],
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
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_"),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_"),
    (
        [960, 8, 3, 3],
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
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_"),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_"),
    (
        [960, 8, 3, 3],
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
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_"),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_"),
    (
        [960, 8, 3, 3],
        "L_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_"),
    (
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_"),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_"),
    (
        [960, 8, 3, 3],
        "L_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_"),
    (
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_"),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_"),
    (
        [960, 8, 3, 3],
        "L_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_"),
    (
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_"),
    (
        [960, 240, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([960], "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_"),
    ([960], "L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_"),
    (
        [960, 8, 3, 3],
        "L_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_",
    ),
    ([240], "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_"),
    ([240], "L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_"),
    (
        [240, 960, 1, 1],
        "L_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 960, 1, 1],
        "L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_",
    ),
    ([960], "L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_"),
    (
        [960, 60, 1, 1],
        "L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [2112, 240, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [2112, 8, 3, 3],
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
        [528, 2112, 1, 1],
        "L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([60], "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_"),
    (
        [60, 2112, 1, 1],
        "L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_",
    ),
    ([2112], "L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_"),
    (
        [2112, 60, 1, 1],
        "L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [2112, 528, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [2112, 8, 3, 3],
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
        [528, 2112, 1, 1],
        "L_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([132], "L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_"),
    (
        [132, 2112, 1, 1],
        "L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_",
    ),
    ([2112], "L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_"),
    (
        [2112, 132, 1, 1],
        "L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [2112, 528, 1, 1],
        "L_self_modules_s4_modules_b3_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [2112, 8, 3, 3],
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
        [528, 2112, 1, 1],
        "L_self_modules_s4_modules_b3_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([132], "L_self_modules_s4_modules_b3_modules_se_modules_fc1_parameters_bias_"),
    (
        [132, 2112, 1, 1],
        "L_self_modules_s4_modules_b3_modules_se_modules_fc1_parameters_weight_",
    ),
    ([2112], "L_self_modules_s4_modules_b3_modules_se_modules_fc2_parameters_bias_"),
    (
        [2112, 132, 1, 1],
        "L_self_modules_s4_modules_b3_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [2112, 528, 1, 1],
        "L_self_modules_s4_modules_b4_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [2112, 8, 3, 3],
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
        [528, 2112, 1, 1],
        "L_self_modules_s4_modules_b4_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([132], "L_self_modules_s4_modules_b4_modules_se_modules_fc1_parameters_bias_"),
    (
        [132, 2112, 1, 1],
        "L_self_modules_s4_modules_b4_modules_se_modules_fc1_parameters_weight_",
    ),
    ([2112], "L_self_modules_s4_modules_b4_modules_se_modules_fc2_parameters_bias_"),
    (
        [2112, 132, 1, 1],
        "L_self_modules_s4_modules_b4_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [2112, 528, 1, 1],
        "L_self_modules_s4_modules_b5_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [2112, 8, 3, 3],
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
        [528, 2112, 1, 1],
        "L_self_modules_s4_modules_b5_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([132], "L_self_modules_s4_modules_b5_modules_se_modules_fc1_parameters_bias_"),
    (
        [132, 2112, 1, 1],
        "L_self_modules_s4_modules_b5_modules_se_modules_fc1_parameters_weight_",
    ),
    ([2112], "L_self_modules_s4_modules_b5_modules_se_modules_fc2_parameters_bias_"),
    (
        [2112, 132, 1, 1],
        "L_self_modules_s4_modules_b5_modules_se_modules_fc2_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_weight_",
    ),
    (
        [2112, 528, 1, 1],
        "L_self_modules_s4_modules_b6_modules_conv1_modules_conv_parameters_weight_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_mean_",
    ),
    (
        [2112],
        "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_var_",
    ),
    ([2112], "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_bias_"),
    (
        [2112],
        "L_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_weight_",
    ),
    (
        [2112, 8, 3, 3],
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
        [528, 2112, 1, 1],
        "L_self_modules_s4_modules_b6_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([132], "L_self_modules_s4_modules_b6_modules_se_modules_fc1_parameters_bias_"),
    (
        [132, 2112, 1, 1],
        "L_self_modules_s4_modules_b6_modules_se_modules_fc1_parameters_weight_",
    ),
    ([2112], "L_self_modules_s4_modules_b6_modules_se_modules_fc2_parameters_bias_"),
    (
        [2112, 132, 1, 1],
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
