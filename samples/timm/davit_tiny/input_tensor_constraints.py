from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 768], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([768], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([768], "L_self_modules_head_modules_norm_parameters_weight_"),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [288, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [96, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [96, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [96, 384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [288, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [96, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [96, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [96, 384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [192, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [192, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [192, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [192, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [192, 96, 2, 2],
        "L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [384, 192, 2, 2],
        "L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [768, 384, 2, 2],
        "L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_",
    ),
    ([96], "L_self_modules_stem_modules_conv_parameters_bias_"),
    ([96, 3, 7, 7], "L_self_modules_stem_modules_conv_parameters_weight_"),
    ([96], "L_self_modules_stem_modules_norm_parameters_bias_"),
    ([96], "L_self_modules_stem_modules_norm_parameters_weight_"),
    ([S0, 3, 224, 224], "L_x_"),
]
