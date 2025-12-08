from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([64], "L_self_modules_bn1_buffers_running_mean_"),
    ([64], "L_self_modules_bn1_buffers_running_var_"),
    ([64], "L_self_modules_bn1_parameters_bias_"),
    ([64], "L_self_modules_bn1_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_conv1_modules_0_parameters_weight_"),
    ([32], "L_self_modules_conv1_modules_1_buffers_running_mean_"),
    ([32], "L_self_modules_conv1_modules_1_buffers_running_var_"),
    ([32], "L_self_modules_conv1_modules_1_parameters_bias_"),
    ([32], "L_self_modules_conv1_modules_1_parameters_weight_"),
    ([32, 32, 3, 3], "L_self_modules_conv1_modules_3_parameters_weight_"),
    ([32], "L_self_modules_conv1_modules_4_buffers_running_mean_"),
    ([32], "L_self_modules_conv1_modules_4_buffers_running_var_"),
    ([32], "L_self_modules_conv1_modules_4_parameters_bias_"),
    ([32], "L_self_modules_conv1_modules_4_parameters_weight_"),
    ([64, 32, 3, 3], "L_self_modules_conv1_modules_6_parameters_weight_"),
    ([1000], "L_self_modules_fc_parameters_bias_"),
    ([1000, 2048], "L_self_modules_fc_parameters_weight_"),
    ([96], "L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_"),
    ([96], "L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_"),
    ([96], "L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_"),
    ([96], "L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_"),
    (
        [96, 64, 1, 1],
        "L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [96, 24, 3, 3],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [32, 24, 1, 1],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [96, 8, 1, 1],
        "L_self_modules_layer1_modules_0_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [256, 96, 1, 1],
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
    ([96], "L_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_"),
    ([96], "L_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_"),
    ([96], "L_self_modules_layer1_modules_1_modules_bn1_parameters_bias_"),
    ([96], "L_self_modules_layer1_modules_1_modules_bn1_parameters_weight_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_parameters_bias_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_parameters_weight_"),
    (
        [96, 256, 1, 1],
        "L_self_modules_layer1_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [96, 24, 3, 3],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [32, 24, 1, 1],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [96, 8, 1, 1],
        "L_self_modules_layer1_modules_1_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [256, 96, 1, 1],
        "L_self_modules_layer1_modules_1_modules_conv3_parameters_weight_",
    ),
    ([96], "L_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_"),
    ([96], "L_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_"),
    ([96], "L_self_modules_layer1_modules_2_modules_bn1_parameters_bias_"),
    ([96], "L_self_modules_layer1_modules_2_modules_bn1_parameters_weight_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_parameters_bias_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_parameters_weight_"),
    (
        [96, 256, 1, 1],
        "L_self_modules_layer1_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [96, 24, 3, 3],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [32, 24, 1, 1],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [96, 8, 1, 1],
        "L_self_modules_layer1_modules_2_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [256, 96, 1, 1],
        "L_self_modules_layer1_modules_2_modules_conv3_parameters_weight_",
    ),
    ([192], "L_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_"),
    ([192], "L_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_"),
    ([192], "L_self_modules_layer2_modules_0_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_layer2_modules_0_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_parameters_weight_"),
    (
        [192, 256, 1, 1],
        "L_self_modules_layer2_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [192, 48, 3, 3],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [48, 48, 1, 1],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [192, 12, 1, 1],
        "L_self_modules_layer2_modules_0_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [512, 192, 1, 1],
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
    ([192], "L_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_"),
    ([192], "L_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_"),
    ([192], "L_self_modules_layer2_modules_1_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_layer2_modules_1_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_parameters_weight_"),
    (
        [192, 512, 1, 1],
        "L_self_modules_layer2_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [192, 48, 3, 3],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [48, 48, 1, 1],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [192, 12, 1, 1],
        "L_self_modules_layer2_modules_1_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [512, 192, 1, 1],
        "L_self_modules_layer2_modules_1_modules_conv3_parameters_weight_",
    ),
    ([192], "L_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_"),
    ([192], "L_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_"),
    ([192], "L_self_modules_layer2_modules_2_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_layer2_modules_2_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_parameters_weight_"),
    (
        [192, 512, 1, 1],
        "L_self_modules_layer2_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [192, 48, 3, 3],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [48, 48, 1, 1],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [192, 12, 1, 1],
        "L_self_modules_layer2_modules_2_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [512, 192, 1, 1],
        "L_self_modules_layer2_modules_2_modules_conv3_parameters_weight_",
    ),
    ([192], "L_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_"),
    ([192], "L_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_"),
    ([192], "L_self_modules_layer2_modules_3_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_layer2_modules_3_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_parameters_weight_"),
    (
        [192, 512, 1, 1],
        "L_self_modules_layer2_modules_3_modules_conv1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [192, 48, 3, 3],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [48, 48, 1, 1],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [192, 12, 1, 1],
        "L_self_modules_layer2_modules_3_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [512, 192, 1, 1],
        "L_self_modules_layer2_modules_3_modules_conv3_parameters_weight_",
    ),
    ([384], "L_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_"),
    ([384], "L_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_"),
    ([384], "L_self_modules_layer3_modules_0_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_layer3_modules_0_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_"),
    (
        [384, 512, 1, 1],
        "L_self_modules_layer3_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [384, 96, 3, 3],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [384, 24, 1, 1],
        "L_self_modules_layer3_modules_0_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [1024, 384, 1, 1],
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
    ([384], "L_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_"),
    ([384], "L_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_"),
    ([384], "L_self_modules_layer3_modules_1_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_layer3_modules_1_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_parameters_weight_"),
    (
        [384, 1024, 1, 1],
        "L_self_modules_layer3_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [384, 96, 3, 3],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [384, 24, 1, 1],
        "L_self_modules_layer3_modules_1_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [1024, 384, 1, 1],
        "L_self_modules_layer3_modules_1_modules_conv3_parameters_weight_",
    ),
    ([384], "L_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_"),
    ([384], "L_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_"),
    ([384], "L_self_modules_layer3_modules_2_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_layer3_modules_2_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_parameters_weight_"),
    (
        [384, 1024, 1, 1],
        "L_self_modules_layer3_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [384, 96, 3, 3],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [384, 24, 1, 1],
        "L_self_modules_layer3_modules_2_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [1024, 384, 1, 1],
        "L_self_modules_layer3_modules_2_modules_conv3_parameters_weight_",
    ),
    ([384], "L_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_"),
    ([384], "L_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_"),
    ([384], "L_self_modules_layer3_modules_3_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_layer3_modules_3_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_parameters_weight_"),
    (
        [384, 1024, 1, 1],
        "L_self_modules_layer3_modules_3_modules_conv1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [384, 96, 3, 3],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [384, 24, 1, 1],
        "L_self_modules_layer3_modules_3_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [1024, 384, 1, 1],
        "L_self_modules_layer3_modules_3_modules_conv3_parameters_weight_",
    ),
    ([384], "L_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_"),
    ([384], "L_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_"),
    ([384], "L_self_modules_layer3_modules_4_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_layer3_modules_4_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_parameters_weight_"),
    (
        [384, 1024, 1, 1],
        "L_self_modules_layer3_modules_4_modules_conv1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [384, 96, 3, 3],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [384, 24, 1, 1],
        "L_self_modules_layer3_modules_4_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [1024, 384, 1, 1],
        "L_self_modules_layer3_modules_4_modules_conv3_parameters_weight_",
    ),
    ([384], "L_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_"),
    ([384], "L_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_"),
    ([384], "L_self_modules_layer3_modules_5_modules_bn1_parameters_bias_"),
    ([384], "L_self_modules_layer3_modules_5_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_parameters_weight_"),
    (
        [384, 1024, 1, 1],
        "L_self_modules_layer3_modules_5_modules_conv1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [384, 96, 3, 3],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [384, 24, 1, 1],
        "L_self_modules_layer3_modules_5_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [1024, 384, 1, 1],
        "L_self_modules_layer3_modules_5_modules_conv3_parameters_weight_",
    ),
    ([768], "L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_"),
    ([768], "L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_"),
    ([768], "L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_parameters_bias_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_parameters_weight_"),
    (
        [768, 1024, 1, 1],
        "L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [768, 192, 3, 3],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [768, 48, 1, 1],
        "L_self_modules_layer4_modules_0_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [2048, 768, 1, 1],
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
    ([768], "L_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_"),
    ([768], "L_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_"),
    ([768], "L_self_modules_layer4_modules_1_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_layer4_modules_1_modules_bn1_parameters_weight_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_parameters_bias_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_parameters_weight_"),
    (
        [768, 2048, 1, 1],
        "L_self_modules_layer4_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [768, 192, 3, 3],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [768, 48, 1, 1],
        "L_self_modules_layer4_modules_1_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [2048, 768, 1, 1],
        "L_self_modules_layer4_modules_1_modules_conv3_parameters_weight_",
    ),
    ([768], "L_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_"),
    ([768], "L_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_"),
    ([768], "L_self_modules_layer4_modules_2_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_layer4_modules_2_modules_bn1_parameters_weight_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_parameters_bias_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_parameters_weight_"),
    (
        [768, 2048, 1, 1],
        "L_self_modules_layer4_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_bn0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_bn1_parameters_weight_",
    ),
    (
        [768, 192, 3, 3],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_bias_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_bias_",
    ),
    (
        [768, 48, 1, 1],
        "L_self_modules_layer4_modules_2_modules_conv2_modules_fc2_parameters_weight_",
    ),
    (
        [2048, 768, 1, 1],
        "L_self_modules_layer4_modules_2_modules_conv3_parameters_weight_",
    ),
    ([1, 3, S0, S0], "L_x_"),
    ([], "s1"),
]
