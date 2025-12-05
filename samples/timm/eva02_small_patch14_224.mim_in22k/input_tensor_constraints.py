from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([384], "L_self_modules_blocks_modules_0_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_0_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_0_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_10_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_10_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_10_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_11_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_11_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_11_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_1_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_1_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_1_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_2_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_3_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_3_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_3_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_4_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_5_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_5_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_5_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_6_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_6_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_6_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_7_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_7_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_7_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_8_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_8_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_8_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_9_modules_attn_buffers_k_bias_"),
    (
        [384],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_9_modules_attn_parameters_q_bias_"),
    ([384], "L_self_modules_blocks_modules_9_modules_attn_parameters_v_bias_"),
    (
        [2048],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 384],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [384, 1024],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([384], "L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_"),
    ([384], "L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_"),
    ([384], "L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_"),
    ([384], "L_self_modules_fc_norm_parameters_bias_"),
    ([384], "L_self_modules_fc_norm_parameters_weight_"),
    ([384], "L_self_modules_patch_embed_modules_proj_parameters_bias_"),
    ([384, 3, 14, 14], "L_self_modules_patch_embed_modules_proj_parameters_weight_"),
    ([256, 128], "L_self_modules_rope_buffers_pos_embed_"),
    ([], "L_self_num_prefix_tokens"),
    ([1, 1, 384], "L_self_parameters_cls_token_"),
    ([1, 257, 384], "L_self_parameters_pos_embed_"),
    ([1, 3, S0, S0], "L_x_"),
]
