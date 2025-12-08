from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [3072],
        "L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_0_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_"),
    (
        [3072],
        "L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_10_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_"),
    (
        [3072],
        "L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_11_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_"),
    (
        [3072],
        "L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_1_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_"),
    (
        [3072],
        "L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_2_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_"),
    (
        [3072],
        "L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_3_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_"),
    (
        [3072],
        "L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_4_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_"),
    (
        [3072],
        "L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_5_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_"),
    (
        [3072],
        "L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_6_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_"),
    (
        [3072],
        "L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_7_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_"),
    (
        [3072],
        "L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_8_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_"),
    (
        [3072],
        "L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_bias_",
    ),
    (
        [384, 196],
        "L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc1_parameters_weight_",
    ),
    (
        [196],
        "L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_bias_",
    ),
    (
        [196, 384],
        "L_self_modules_blocks_modules_9_modules_mlp_tokens_modules_fc2_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_"),
    ([21843], "L_self_modules_head_parameters_bias_"),
    ([21843, 768], "L_self_modules_head_parameters_weight_"),
    ([768], "L_self_modules_norm_parameters_bias_"),
    ([768], "L_self_modules_norm_parameters_weight_"),
    ([768], "L_self_modules_stem_modules_proj_parameters_bias_"),
    ([768, 3, 16, 16], "L_self_modules_stem_modules_proj_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
]
