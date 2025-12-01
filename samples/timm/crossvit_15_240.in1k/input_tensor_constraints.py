from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 576],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [192, 384],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [192, 384],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 576],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [192, 384],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [192, 384],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 576],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1152],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [192, 384],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [192, 384],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [384, 192],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_weight_",
    ),
    ([1000], "L_self_modules_head_modules_0_parameters_bias_"),
    ([1000, 192], "L_self_modules_head_modules_0_parameters_weight_"),
    ([1000], "L_self_modules_head_modules_1_parameters_bias_"),
    ([1000, 384], "L_self_modules_head_modules_1_parameters_weight_"),
    ([192], "L_self_modules_norm_modules_0_parameters_bias_"),
    ([192], "L_self_modules_norm_modules_0_parameters_weight_"),
    ([384], "L_self_modules_norm_modules_1_parameters_bias_"),
    ([384], "L_self_modules_norm_modules_1_parameters_weight_"),
    ([192], "L_self_modules_patch_embed_modules_0_modules_proj_parameters_bias_"),
    (
        [192, 3, 12, 12],
        "L_self_modules_patch_embed_modules_0_modules_proj_parameters_weight_",
    ),
    ([384], "L_self_modules_patch_embed_modules_1_modules_proj_parameters_bias_"),
    (
        [384, 3, 16, 16],
        "L_self_modules_patch_embed_modules_1_modules_proj_parameters_weight_",
    ),
    ([1, 1, 192], "L_self_parameters_cls_token_0_"),
    ([1, 1, 384], "L_self_parameters_cls_token_1_"),
    ([1, 401, 192], "L_self_parameters_pos_embed_0_"),
    ([1, 197, 384], "L_self_parameters_pos_embed_1_"),
    ([1, 3, S0, S0], "L_x_"),
]
