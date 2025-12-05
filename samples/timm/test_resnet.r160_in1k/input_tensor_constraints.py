from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([32], "L_self_modules_bn1_buffers_running_mean_"),
    ([32], "L_self_modules_bn1_buffers_running_var_"),
    ([32], "L_self_modules_bn1_parameters_bias_"),
    ([32], "L_self_modules_bn1_parameters_weight_"),
    ([16, 3, 3, 3], "L_self_modules_conv1_modules_0_parameters_weight_"),
    ([16], "L_self_modules_conv1_modules_1_buffers_running_mean_"),
    ([16], "L_self_modules_conv1_modules_1_buffers_running_var_"),
    ([16], "L_self_modules_conv1_modules_1_parameters_bias_"),
    ([16], "L_self_modules_conv1_modules_1_parameters_weight_"),
    ([16, 16, 3, 3], "L_self_modules_conv1_modules_3_parameters_weight_"),
    ([16], "L_self_modules_conv1_modules_4_buffers_running_mean_"),
    ([16], "L_self_modules_conv1_modules_4_buffers_running_var_"),
    ([16], "L_self_modules_conv1_modules_4_parameters_bias_"),
    ([16], "L_self_modules_conv1_modules_4_parameters_weight_"),
    ([32, 16, 3, 3], "L_self_modules_conv1_modules_6_parameters_weight_"),
    ([1000], "L_self_modules_fc_parameters_bias_"),
    ([1000, 96], "L_self_modules_fc_parameters_weight_"),
    ([32], "L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_"),
    ([32], "L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_"),
    ([32], "L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_"),
    ([32], "L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_"),
    ([32], "L_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_"),
    ([32], "L_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_"),
    ([32], "L_self_modules_layer1_modules_0_modules_bn2_parameters_bias_"),
    ([32], "L_self_modules_layer1_modules_0_modules_bn2_parameters_weight_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_layer1_modules_0_modules_conv2_parameters_weight_",
    ),
    ([48], "L_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_"),
    ([48], "L_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_"),
    ([48], "L_self_modules_layer2_modules_0_modules_bn1_parameters_bias_"),
    ([48], "L_self_modules_layer2_modules_0_modules_bn1_parameters_weight_"),
    ([48], "L_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_"),
    ([48], "L_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_"),
    ([48], "L_self_modules_layer2_modules_0_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_layer2_modules_0_modules_bn2_parameters_weight_"),
    (
        [48, 32, 3, 3],
        "L_self_modules_layer2_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [48, 48, 3, 3],
        "L_self_modules_layer2_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [48, 32, 1, 1],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_",
    ),
    ([48], "L_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_"),
    ([48], "L_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_"),
    ([48], "L_self_modules_layer3_modules_0_modules_bn1_parameters_bias_"),
    ([48], "L_self_modules_layer3_modules_0_modules_bn1_parameters_weight_"),
    ([48], "L_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_"),
    ([48], "L_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_"),
    ([48], "L_self_modules_layer3_modules_0_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_layer3_modules_0_modules_bn2_parameters_weight_"),
    ([192], "L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_"),
    ([192], "L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_"),
    ([192], "L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_"),
    (
        [48, 48, 1, 1],
        "L_self_modules_layer3_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [48, 48, 3, 3],
        "L_self_modules_layer3_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [192, 48, 1, 1],
        "L_self_modules_layer3_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [192, 48, 1, 1],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_",
    ),
    ([96], "L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_"),
    ([96], "L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_"),
    ([96], "L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_"),
    ([96], "L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_"),
    ([96], "L_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_"),
    ([96], "L_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_"),
    ([96], "L_self_modules_layer4_modules_0_modules_bn2_parameters_bias_"),
    ([96], "L_self_modules_layer4_modules_0_modules_bn2_parameters_weight_"),
    (
        [96, 192, 3, 3],
        "L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_layer4_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [96, 192, 1, 1],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_",
    ),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
