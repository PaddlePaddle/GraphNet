from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 640], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([640], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([640], "L_self_modules_head_modules_norm_parameters_weight_"),
    (
        [80],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_",
    ),
    (
        [80, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [320, 80, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [80, 320, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_",
    ),
    (
        [80, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [320, 80, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [80, 320, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_",
    ),
    (
        [160, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [640, 160, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_",
    ),
    (
        [160, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [640, 160, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [160, 80, 2, 2],
        "L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_",
    ),
    (
        [320, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_bias_",
    ),
    (
        [320, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_bias_",
    ),
    (
        [320, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_bias_",
    ),
    (
        [320, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_bias_",
    ),
    (
        [320, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_bias_",
    ),
    (
        [320, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_bias_",
    ),
    (
        [320, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_parameters_gamma_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_bias_",
    ),
    (
        [320, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_dw_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1280, 320, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 1280, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_parameters_gamma_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [320, 160, 2, 2],
        "L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_",
    ),
    (
        [640, 1, 7, 7],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2560, 640, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [640, 2560, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_bias_",
    ),
    (
        [640, 1, 7, 7],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2560, 640, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [640, 2560, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [640, 320, 2, 2],
        "L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_",
    ),
    ([80], "L_self_modules_stem_modules_0_parameters_bias_"),
    ([80, 3, 3, 3], "L_self_modules_stem_modules_0_parameters_weight_"),
    ([80], "L_self_modules_stem_modules_1_parameters_bias_"),
    ([80, 80, 3, 3], "L_self_modules_stem_modules_1_parameters_weight_"),
    ([80], "L_self_modules_stem_modules_2_parameters_bias_"),
    ([80], "L_self_modules_stem_modules_2_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
