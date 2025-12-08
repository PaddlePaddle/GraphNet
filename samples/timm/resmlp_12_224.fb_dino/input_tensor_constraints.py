from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([196], "L_self_modules_blocks_modules_0_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_0_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_0_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_0_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_0_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_0_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_0_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_0_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_0_parameters_ls2_"),
    ([196], "L_self_modules_blocks_modules_10_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_10_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_10_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_10_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_10_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_10_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_10_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_10_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_10_parameters_ls2_"),
    ([196], "L_self_modules_blocks_modules_11_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_11_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_11_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_11_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_11_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_11_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_11_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_11_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_11_parameters_ls2_"),
    ([196], "L_self_modules_blocks_modules_1_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_1_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_1_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_1_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_1_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_1_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_1_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_1_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_1_parameters_ls2_"),
    ([196], "L_self_modules_blocks_modules_2_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_2_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_2_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_2_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_2_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_2_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_2_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_2_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_2_parameters_ls2_"),
    ([196], "L_self_modules_blocks_modules_3_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_3_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_3_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_3_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_3_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_3_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_3_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_3_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_3_parameters_ls2_"),
    ([196], "L_self_modules_blocks_modules_4_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_4_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_4_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_4_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_4_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_4_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_4_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_4_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_4_parameters_ls2_"),
    ([196], "L_self_modules_blocks_modules_5_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_5_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_5_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_5_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_5_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_5_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_5_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_5_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_5_parameters_ls2_"),
    ([196], "L_self_modules_blocks_modules_6_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_6_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_6_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_6_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_6_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_6_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_6_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_6_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_6_parameters_ls2_"),
    ([196], "L_self_modules_blocks_modules_7_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_7_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_7_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_7_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_7_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_7_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_7_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_7_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_7_parameters_ls2_"),
    ([196], "L_self_modules_blocks_modules_8_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_8_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_8_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_8_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_8_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_8_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_8_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_8_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_8_parameters_ls2_"),
    ([196], "L_self_modules_blocks_modules_9_modules_linear_tokens_parameters_bias_"),
    (
        [196, 196],
        "L_self_modules_blocks_modules_9_modules_linear_tokens_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_9_modules_mlp_channels_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_blocks_modules_9_modules_norm1_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_9_modules_norm1_parameters_beta_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_9_modules_norm2_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_blocks_modules_9_modules_norm2_parameters_beta_"),
    ([384], "L_self_modules_blocks_modules_9_parameters_ls1_"),
    ([384], "L_self_modules_blocks_modules_9_parameters_ls2_"),
    ([1000], "L_self_modules_head_parameters_bias_"),
    ([1000, 384], "L_self_modules_head_parameters_weight_"),
    ([1, 1, 384], "L_self_modules_norm_parameters_alpha_"),
    ([1, 1, 384], "L_self_modules_norm_parameters_beta_"),
    ([384], "L_self_modules_stem_modules_proj_parameters_bias_"),
    ([384, 3, 16, 16], "L_self_modules_stem_modules_proj_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
]
