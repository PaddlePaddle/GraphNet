from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [4],
        "L_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_bias_",
    ),
    (
        [4, 3],
        "L_self_modules_blocks_modules_0_modules_attn_modules_pos_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_0_modules_attn_modules_qk_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_0_modules_attn_modules_v_parameters_weight_",
    ),
    ([4], "L_self_modules_blocks_modules_0_modules_attn_parameters_gating_param_"),
    ([768], "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_"),
    (
        [4],
        "L_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_bias_",
    ),
    (
        [4, 3],
        "L_self_modules_blocks_modules_1_modules_attn_modules_pos_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_1_modules_attn_modules_qk_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_1_modules_attn_modules_v_parameters_weight_",
    ),
    ([4], "L_self_modules_blocks_modules_1_modules_attn_parameters_gating_param_"),
    ([768], "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_"),
    (
        [4],
        "L_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_bias_",
    ),
    (
        [4, 3],
        "L_self_modules_blocks_modules_2_modules_attn_modules_pos_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_2_modules_attn_modules_qk_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_2_modules_attn_modules_v_parameters_weight_",
    ),
    ([4], "L_self_modules_blocks_modules_2_modules_attn_parameters_gating_param_"),
    ([768], "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_"),
    (
        [4],
        "L_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_bias_",
    ),
    (
        [4, 3],
        "L_self_modules_blocks_modules_3_modules_attn_modules_pos_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_3_modules_attn_modules_qk_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_3_modules_attn_modules_v_parameters_weight_",
    ),
    ([4], "L_self_modules_blocks_modules_3_modules_attn_parameters_gating_param_"),
    ([768], "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_"),
    (
        [4],
        "L_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_bias_",
    ),
    (
        [4, 3],
        "L_self_modules_blocks_modules_4_modules_attn_modules_pos_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_4_modules_attn_modules_qk_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_4_modules_attn_modules_v_parameters_weight_",
    ),
    ([4], "L_self_modules_blocks_modules_4_modules_attn_parameters_gating_param_"),
    ([768], "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_"),
    (
        [4],
        "L_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_bias_",
    ),
    (
        [4, 3],
        "L_self_modules_blocks_modules_5_modules_attn_modules_pos_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_5_modules_attn_modules_qk_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_5_modules_attn_modules_v_parameters_weight_",
    ),
    ([4], "L_self_modules_blocks_modules_5_modules_attn_parameters_gating_param_"),
    ([768], "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_"),
    (
        [4],
        "L_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_bias_",
    ),
    (
        [4, 3],
        "L_self_modules_blocks_modules_6_modules_attn_modules_pos_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_6_modules_attn_modules_qk_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_6_modules_attn_modules_v_parameters_weight_",
    ),
    ([4], "L_self_modules_blocks_modules_6_modules_attn_parameters_gating_param_"),
    ([768], "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_"),
    (
        [4],
        "L_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_bias_",
    ),
    (
        [4, 3],
        "L_self_modules_blocks_modules_7_modules_attn_modules_pos_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_7_modules_attn_modules_qk_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_7_modules_attn_modules_v_parameters_weight_",
    ),
    ([4], "L_self_modules_blocks_modules_7_modules_attn_parameters_gating_param_"),
    ([768], "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_"),
    (
        [4],
        "L_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_bias_",
    ),
    (
        [4, 3],
        "L_self_modules_blocks_modules_8_modules_attn_modules_pos_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_8_modules_attn_modules_qk_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_8_modules_attn_modules_v_parameters_weight_",
    ),
    ([4], "L_self_modules_blocks_modules_8_modules_attn_parameters_gating_param_"),
    ([768], "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_"),
    (
        [4],
        "L_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_bias_",
    ),
    (
        [4, 3],
        "L_self_modules_blocks_modules_9_modules_attn_modules_pos_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_9_modules_attn_modules_qk_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_9_modules_attn_modules_v_parameters_weight_",
    ),
    ([4], "L_self_modules_blocks_modules_9_modules_attn_parameters_gating_param_"),
    ([768], "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_"),
    ([1000], "L_self_modules_head_parameters_bias_"),
    ([1000, 192], "L_self_modules_head_parameters_weight_"),
    ([192], "L_self_modules_norm_parameters_bias_"),
    ([192], "L_self_modules_norm_parameters_weight_"),
    ([192], "L_self_modules_patch_embed_modules_proj_parameters_bias_"),
    ([192, 3, 16, 16], "L_self_modules_patch_embed_modules_proj_parameters_weight_"),
    ([1, 1, 192], "L_self_parameters_cls_token_"),
    ([1, 196, 192], "L_self_parameters_pos_embed_"),
    ([1, 3, S0, S0], "L_x_"),
]
