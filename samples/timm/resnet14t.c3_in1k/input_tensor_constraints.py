from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([64], "L_self_modules_bn1_buffers_running_mean_"),
    ([64], "L_self_modules_bn1_buffers_running_var_"),
    ([64], "L_self_modules_bn1_parameters_bias_"),
    ([64], "L_self_modules_bn1_parameters_weight_"),
    ([24, 3, 3, 3], "L_self_modules_conv1_modules_0_parameters_weight_"),
    ([24], "L_self_modules_conv1_modules_1_buffers_running_mean_"),
    ([24], "L_self_modules_conv1_modules_1_buffers_running_var_"),
    ([24], "L_self_modules_conv1_modules_1_parameters_bias_"),
    ([24], "L_self_modules_conv1_modules_1_parameters_weight_"),
    ([32, 24, 3, 3], "L_self_modules_conv1_modules_3_parameters_weight_"),
    ([32], "L_self_modules_conv1_modules_4_buffers_running_mean_"),
    ([32], "L_self_modules_conv1_modules_4_buffers_running_var_"),
    ([32], "L_self_modules_conv1_modules_4_parameters_bias_"),
    ([32], "L_self_modules_conv1_modules_4_parameters_weight_"),
    ([64, 32, 3, 3], "L_self_modules_conv1_modules_6_parameters_weight_"),
    ([1000], "L_self_modules_fc_parameters_bias_"),
    ([1000, 2048], "L_self_modules_fc_parameters_weight_"),
    ([64], "L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_"),
    ([64], "L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_"),
    ([64], "L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_"),
    ([64], "L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_"),
    ([64], "L_self_modules_layer1_modules_0_modules_bn2_buffers_running_mean_"),
    ([64], "L_self_modules_layer1_modules_0_modules_bn2_buffers_running_var_"),
    ([64], "L_self_modules_layer1_modules_0_modules_bn2_parameters_bias_"),
    ([64], "L_self_modules_layer1_modules_0_modules_bn2_parameters_weight_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_"),
    (
        [64, 64, 1, 1],
        "L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_layer1_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_",
    ),
    ([128], "L_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_"),
    ([128], "L_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_"),
    ([128], "L_self_modules_layer2_modules_0_modules_bn1_parameters_bias_"),
    ([128], "L_self_modules_layer2_modules_0_modules_bn1_parameters_weight_"),
    ([128], "L_self_modules_layer2_modules_0_modules_bn2_buffers_running_mean_"),
    ([128], "L_self_modules_layer2_modules_0_modules_bn2_buffers_running_var_"),
    ([128], "L_self_modules_layer2_modules_0_modules_bn2_parameters_bias_"),
    ([128], "L_self_modules_layer2_modules_0_modules_bn2_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_parameters_weight_"),
    (
        [128, 256, 1, 1],
        "L_self_modules_layer2_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_layer2_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_layer2_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_",
    ),
    ([256], "L_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_"),
    ([256], "L_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_"),
    ([256], "L_self_modules_layer3_modules_0_modules_bn1_parameters_bias_"),
    ([256], "L_self_modules_layer3_modules_0_modules_bn1_parameters_weight_"),
    ([256], "L_self_modules_layer3_modules_0_modules_bn2_buffers_running_mean_"),
    ([256], "L_self_modules_layer3_modules_0_modules_bn2_buffers_running_var_"),
    ([256], "L_self_modules_layer3_modules_0_modules_bn2_parameters_bias_"),
    ([256], "L_self_modules_layer3_modules_0_modules_bn2_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_"),
    (
        [256, 512, 1, 1],
        "L_self_modules_layer3_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_layer3_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_layer3_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [1024, 512, 1, 1],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_",
    ),
    ([512], "L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_"),
    ([512], "L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_"),
    ([512], "L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_"),
    ([512], "L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer4_modules_0_modules_bn2_buffers_running_mean_"),
    ([512], "L_self_modules_layer4_modules_0_modules_bn2_buffers_running_var_"),
    ([512], "L_self_modules_layer4_modules_0_modules_bn2_parameters_bias_"),
    ([512], "L_self_modules_layer4_modules_0_modules_bn2_parameters_weight_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_parameters_bias_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_parameters_weight_"),
    (
        [512, 1024, 1, 1],
        "L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_layer4_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layer4_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [2048, 1024, 1, 1],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_",
    ),
    (
        [2048],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_",
    ),
    (
        [2048],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_",
    ),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
