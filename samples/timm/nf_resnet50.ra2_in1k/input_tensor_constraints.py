from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 2048], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([64], "L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_gain_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_",
    ),
    ([64], "L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_gain_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_gain_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_gain_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([64], "L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_",
    ),
    ([64], "L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_",
    ),
    ([64], "L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_gain_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_stages_modules_0_modules_2_modules_conv1_parameters_weight_",
    ),
    ([64], "L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_gain_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_0_modules_2_modules_conv2_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_gain_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_2_modules_conv3_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_",
    ),
    (
        [256, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_",
    ),
    (
        [1024, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [1024, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_",
    ),
    (
        [1024, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_",
    ),
    (
        [1024, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_",
    ),
    (
        [1024, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_",
    ),
    (
        [1024, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_",
    ),
    (
        [1024, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_",
    ),
    (
        [1024, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_",
    ),
    (
        [512, 1024, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_",
    ),
    (
        [2048, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [2048, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_",
    ),
    (
        [2048, 1024, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_",
    ),
    (
        [2048, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_",
    ),
    (
        [2048, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_",
    ),
    ([64], "L_self_modules_stem_modules_conv_parameters_bias_"),
    ([64, 1, 1, 1], "L_self_modules_stem_modules_conv_parameters_gain_"),
    ([64, 3, 7, 7], "L_self_modules_stem_modules_conv_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
