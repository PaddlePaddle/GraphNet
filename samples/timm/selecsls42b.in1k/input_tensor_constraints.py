from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_fc_parameters_bias_"),
    ([1000, 1024], "L_self_modules_fc_parameters_weight_"),
    (
        [64, 32, 3, 3],
        "L_self_modules_features_modules_0_modules_conv1_modules_0_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv1_modules_1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv1_modules_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv1_modules_1_parameters_weight_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_features_modules_0_modules_conv2_modules_0_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv2_modules_1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv2_modules_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv2_modules_1_parameters_weight_",
    ),
    (
        [32, 64, 3, 3],
        "L_self_modules_features_modules_0_modules_conv3_modules_0_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_features_modules_0_modules_conv3_modules_1_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_features_modules_0_modules_conv3_modules_1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_features_modules_0_modules_conv3_modules_1_parameters_weight_",
    ),
    (
        [64, 32, 1, 1],
        "L_self_modules_features_modules_0_modules_conv4_modules_0_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv4_modules_1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv4_modules_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv4_modules_1_parameters_weight_",
    ),
    (
        [32, 64, 3, 3],
        "L_self_modules_features_modules_0_modules_conv5_modules_0_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_features_modules_0_modules_conv5_modules_1_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_features_modules_0_modules_conv5_modules_1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_features_modules_0_modules_conv5_modules_1_parameters_weight_",
    ),
    (
        [64, 128, 1, 1],
        "L_self_modules_features_modules_0_modules_conv6_modules_0_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv6_modules_1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv6_modules_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_features_modules_0_modules_conv6_modules_1_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_features_modules_1_modules_conv1_modules_0_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv1_modules_1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv1_modules_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv1_modules_1_parameters_weight_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_features_modules_1_modules_conv2_modules_0_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv2_modules_1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv2_modules_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv2_modules_1_parameters_weight_",
    ),
    (
        [32, 64, 3, 3],
        "L_self_modules_features_modules_1_modules_conv3_modules_0_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_features_modules_1_modules_conv3_modules_1_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_features_modules_1_modules_conv3_modules_1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_features_modules_1_modules_conv3_modules_1_parameters_weight_",
    ),
    (
        [64, 32, 1, 1],
        "L_self_modules_features_modules_1_modules_conv4_modules_0_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv4_modules_1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv4_modules_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_features_modules_1_modules_conv4_modules_1_parameters_weight_",
    ),
    (
        [32, 64, 3, 3],
        "L_self_modules_features_modules_1_modules_conv5_modules_0_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_features_modules_1_modules_conv5_modules_1_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_features_modules_1_modules_conv5_modules_1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_features_modules_1_modules_conv5_modules_1_parameters_weight_",
    ),
    (
        [128, 192, 1, 1],
        "L_self_modules_features_modules_1_modules_conv6_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_features_modules_1_modules_conv6_modules_1_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_features_modules_1_modules_conv6_modules_1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_features_modules_1_modules_conv6_modules_1_parameters_weight_",
    ),
    (
        [144, 128, 3, 3],
        "L_self_modules_features_modules_2_modules_conv1_modules_0_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv1_modules_1_buffers_running_var_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv1_modules_1_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv1_modules_1_parameters_weight_",
    ),
    (
        [144, 144, 1, 1],
        "L_self_modules_features_modules_2_modules_conv2_modules_0_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv2_modules_1_buffers_running_var_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv2_modules_1_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv2_modules_1_parameters_weight_",
    ),
    (
        [72, 144, 3, 3],
        "L_self_modules_features_modules_2_modules_conv3_modules_0_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_features_modules_2_modules_conv3_modules_1_buffers_running_var_",
    ),
    (
        [72],
        "L_self_modules_features_modules_2_modules_conv3_modules_1_parameters_bias_",
    ),
    (
        [72],
        "L_self_modules_features_modules_2_modules_conv3_modules_1_parameters_weight_",
    ),
    (
        [144, 72, 1, 1],
        "L_self_modules_features_modules_2_modules_conv4_modules_0_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv4_modules_1_buffers_running_var_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv4_modules_1_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv4_modules_1_parameters_weight_",
    ),
    (
        [72, 144, 3, 3],
        "L_self_modules_features_modules_2_modules_conv5_modules_0_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_features_modules_2_modules_conv5_modules_1_buffers_running_var_",
    ),
    (
        [72],
        "L_self_modules_features_modules_2_modules_conv5_modules_1_parameters_bias_",
    ),
    (
        [72],
        "L_self_modules_features_modules_2_modules_conv5_modules_1_parameters_weight_",
    ),
    (
        [144, 288, 1, 1],
        "L_self_modules_features_modules_2_modules_conv6_modules_0_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv6_modules_1_buffers_running_var_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv6_modules_1_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_features_modules_2_modules_conv6_modules_1_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_features_modules_3_modules_conv1_modules_0_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv1_modules_1_buffers_running_var_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv1_modules_1_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv1_modules_1_parameters_weight_",
    ),
    (
        [144, 144, 1, 1],
        "L_self_modules_features_modules_3_modules_conv2_modules_0_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv2_modules_1_buffers_running_var_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv2_modules_1_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv2_modules_1_parameters_weight_",
    ),
    (
        [72, 144, 3, 3],
        "L_self_modules_features_modules_3_modules_conv3_modules_0_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_features_modules_3_modules_conv3_modules_1_buffers_running_var_",
    ),
    (
        [72],
        "L_self_modules_features_modules_3_modules_conv3_modules_1_parameters_bias_",
    ),
    (
        [72],
        "L_self_modules_features_modules_3_modules_conv3_modules_1_parameters_weight_",
    ),
    (
        [144, 72, 1, 1],
        "L_self_modules_features_modules_3_modules_conv4_modules_0_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv4_modules_1_buffers_running_var_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv4_modules_1_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_features_modules_3_modules_conv4_modules_1_parameters_weight_",
    ),
    (
        [72, 144, 3, 3],
        "L_self_modules_features_modules_3_modules_conv5_modules_0_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_mean_",
    ),
    (
        [72],
        "L_self_modules_features_modules_3_modules_conv5_modules_1_buffers_running_var_",
    ),
    (
        [72],
        "L_self_modules_features_modules_3_modules_conv5_modules_1_parameters_bias_",
    ),
    (
        [72],
        "L_self_modules_features_modules_3_modules_conv5_modules_1_parameters_weight_",
    ),
    (
        [288, 432, 1, 1],
        "L_self_modules_features_modules_3_modules_conv6_modules_0_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_mean_",
    ),
    (
        [288],
        "L_self_modules_features_modules_3_modules_conv6_modules_1_buffers_running_var_",
    ),
    (
        [288],
        "L_self_modules_features_modules_3_modules_conv6_modules_1_parameters_bias_",
    ),
    (
        [288],
        "L_self_modules_features_modules_3_modules_conv6_modules_1_parameters_weight_",
    ),
    (
        [304, 288, 3, 3],
        "L_self_modules_features_modules_4_modules_conv1_modules_0_parameters_weight_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_mean_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv1_modules_1_buffers_running_var_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv1_modules_1_parameters_bias_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv1_modules_1_parameters_weight_",
    ),
    (
        [304, 304, 1, 1],
        "L_self_modules_features_modules_4_modules_conv2_modules_0_parameters_weight_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_mean_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv2_modules_1_buffers_running_var_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv2_modules_1_parameters_bias_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv2_modules_1_parameters_weight_",
    ),
    (
        [152, 304, 3, 3],
        "L_self_modules_features_modules_4_modules_conv3_modules_0_parameters_weight_",
    ),
    (
        [152],
        "L_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_features_modules_4_modules_conv3_modules_1_buffers_running_var_",
    ),
    (
        [152],
        "L_self_modules_features_modules_4_modules_conv3_modules_1_parameters_bias_",
    ),
    (
        [152],
        "L_self_modules_features_modules_4_modules_conv3_modules_1_parameters_weight_",
    ),
    (
        [304, 152, 1, 1],
        "L_self_modules_features_modules_4_modules_conv4_modules_0_parameters_weight_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_mean_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv4_modules_1_buffers_running_var_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv4_modules_1_parameters_bias_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv4_modules_1_parameters_weight_",
    ),
    (
        [152, 304, 3, 3],
        "L_self_modules_features_modules_4_modules_conv5_modules_0_parameters_weight_",
    ),
    (
        [152],
        "L_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_features_modules_4_modules_conv5_modules_1_buffers_running_var_",
    ),
    (
        [152],
        "L_self_modules_features_modules_4_modules_conv5_modules_1_parameters_bias_",
    ),
    (
        [152],
        "L_self_modules_features_modules_4_modules_conv5_modules_1_parameters_weight_",
    ),
    (
        [304, 608, 1, 1],
        "L_self_modules_features_modules_4_modules_conv6_modules_0_parameters_weight_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_mean_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv6_modules_1_buffers_running_var_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv6_modules_1_parameters_bias_",
    ),
    (
        [304],
        "L_self_modules_features_modules_4_modules_conv6_modules_1_parameters_weight_",
    ),
    (
        [304, 304, 3, 3],
        "L_self_modules_features_modules_5_modules_conv1_modules_0_parameters_weight_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_mean_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv1_modules_1_buffers_running_var_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv1_modules_1_parameters_bias_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv1_modules_1_parameters_weight_",
    ),
    (
        [304, 304, 1, 1],
        "L_self_modules_features_modules_5_modules_conv2_modules_0_parameters_weight_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_mean_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv2_modules_1_buffers_running_var_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv2_modules_1_parameters_bias_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv2_modules_1_parameters_weight_",
    ),
    (
        [152, 304, 3, 3],
        "L_self_modules_features_modules_5_modules_conv3_modules_0_parameters_weight_",
    ),
    (
        [152],
        "L_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_features_modules_5_modules_conv3_modules_1_buffers_running_var_",
    ),
    (
        [152],
        "L_self_modules_features_modules_5_modules_conv3_modules_1_parameters_bias_",
    ),
    (
        [152],
        "L_self_modules_features_modules_5_modules_conv3_modules_1_parameters_weight_",
    ),
    (
        [304, 152, 1, 1],
        "L_self_modules_features_modules_5_modules_conv4_modules_0_parameters_weight_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_mean_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv4_modules_1_buffers_running_var_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv4_modules_1_parameters_bias_",
    ),
    (
        [304],
        "L_self_modules_features_modules_5_modules_conv4_modules_1_parameters_weight_",
    ),
    (
        [152, 304, 3, 3],
        "L_self_modules_features_modules_5_modules_conv5_modules_0_parameters_weight_",
    ),
    (
        [152],
        "L_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_features_modules_5_modules_conv5_modules_1_buffers_running_var_",
    ),
    (
        [152],
        "L_self_modules_features_modules_5_modules_conv5_modules_1_parameters_bias_",
    ),
    (
        [152],
        "L_self_modules_features_modules_5_modules_conv5_modules_1_parameters_weight_",
    ),
    (
        [480, 912, 1, 1],
        "L_self_modules_features_modules_5_modules_conv6_modules_0_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_mean_",
    ),
    (
        [480],
        "L_self_modules_features_modules_5_modules_conv6_modules_1_buffers_running_var_",
    ),
    (
        [480],
        "L_self_modules_features_modules_5_modules_conv6_modules_1_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_features_modules_5_modules_conv6_modules_1_parameters_weight_",
    ),
    ([960, 480, 3, 3], "L_self_modules_head_modules_0_modules_0_parameters_weight_"),
    ([960], "L_self_modules_head_modules_0_modules_1_buffers_running_mean_"),
    ([960], "L_self_modules_head_modules_0_modules_1_buffers_running_var_"),
    ([960], "L_self_modules_head_modules_0_modules_1_parameters_bias_"),
    ([960], "L_self_modules_head_modules_0_modules_1_parameters_weight_"),
    ([1024, 960, 3, 3], "L_self_modules_head_modules_1_modules_0_parameters_weight_"),
    ([1024], "L_self_modules_head_modules_1_modules_1_buffers_running_mean_"),
    ([1024], "L_self_modules_head_modules_1_modules_1_buffers_running_var_"),
    ([1024], "L_self_modules_head_modules_1_modules_1_parameters_bias_"),
    ([1024], "L_self_modules_head_modules_1_modules_1_parameters_weight_"),
    ([1280, 1024, 3, 3], "L_self_modules_head_modules_2_modules_0_parameters_weight_"),
    ([1280], "L_self_modules_head_modules_2_modules_1_buffers_running_mean_"),
    ([1280], "L_self_modules_head_modules_2_modules_1_buffers_running_var_"),
    ([1280], "L_self_modules_head_modules_2_modules_1_parameters_bias_"),
    ([1280], "L_self_modules_head_modules_2_modules_1_parameters_weight_"),
    ([1024, 1280, 1, 1], "L_self_modules_head_modules_3_modules_0_parameters_weight_"),
    ([1024], "L_self_modules_head_modules_3_modules_1_buffers_running_mean_"),
    ([1024], "L_self_modules_head_modules_3_modules_1_buffers_running_var_"),
    ([1024], "L_self_modules_head_modules_3_modules_1_parameters_bias_"),
    ([1024], "L_self_modules_head_modules_3_modules_1_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_stem_modules_0_parameters_weight_"),
    ([32], "L_self_modules_stem_modules_1_buffers_running_mean_"),
    ([32], "L_self_modules_stem_modules_1_buffers_running_var_"),
    ([32], "L_self_modules_stem_modules_1_parameters_bias_"),
    ([32], "L_self_modules_stem_modules_1_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
