from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_0_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_0_parameters_gamma_2_"),
    (
        [384],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_10_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_10_parameters_gamma_2_"),
    (
        [384],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_11_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_11_parameters_gamma_2_"),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_1_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_1_parameters_gamma_2_"),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_2_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_2_parameters_gamma_2_"),
    (
        [384],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_3_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_3_parameters_gamma_2_"),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_4_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_4_parameters_gamma_2_"),
    (
        [384],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_5_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_5_parameters_gamma_2_"),
    (
        [384],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_6_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_6_parameters_gamma_2_"),
    (
        [384],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_7_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_7_parameters_gamma_2_"),
    (
        [384],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_8_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_8_parameters_gamma_2_"),
    (
        [384],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1536],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_9_parameters_gamma_1_"),
    ([384], "L_self_modules_blocks_modules_9_parameters_gamma_2_"),
    ([1000], "L_self_modules_head_parameters_bias_"),
    ([1000, 384], "L_self_modules_head_parameters_weight_"),
    ([384], "L_self_modules_norm_parameters_bias_"),
    ([384], "L_self_modules_norm_parameters_weight_"),
    ([384], "L_self_modules_patch_embed_modules_proj_parameters_bias_"),
    ([384, 3, 16, 16], "L_self_modules_patch_embed_modules_proj_parameters_weight_"),
    ([196], "L_self_modules_rope_buffers_t_x_"),
    ([196], "L_self_modules_rope_buffers_t_y_"),
    ([2, 12, 6, 32], "L_self_modules_rope_parameters_freqs_"),
    ([1, 1, 384], "L_self_parameters_cls_token_"),
    ([1, 3, S0, S0], "L_x_"),
]
