from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 128], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([128], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([128], "L_self_modules_head_modules_norm_parameters_weight_"),
    (
        [32],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_",
    ),
    (
        [32, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [128, 32],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [32, 128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_",
    ),
    (
        [64, 1, 5, 5],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [256, 64],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [32],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [64, 32, 2, 2],
        "L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_bias_",
    ),
    (
        [96, 1, 5, 5],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 96],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [96, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [96, 64, 2, 2],
        "L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_bias_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [128, 96, 2, 2],
        "L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_",
    ),
    ([32], "L_self_modules_stem_modules_0_parameters_bias_"),
    ([32, 3, 4, 4], "L_self_modules_stem_modules_0_parameters_weight_"),
    ([32], "L_self_modules_stem_modules_1_parameters_bias_"),
    ([32], "L_self_modules_stem_modules_1_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
