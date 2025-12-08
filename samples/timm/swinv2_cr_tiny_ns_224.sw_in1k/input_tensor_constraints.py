from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 768], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([96], "L_self_modules_patch_embed_modules_norm_parameters_bias_"),
    ([96], "L_self_modules_patch_embed_modules_norm_parameters_weight_"),
    ([96], "L_self_modules_patch_embed_modules_proj_parameters_bias_"),
    ([96, 3, 4, 4], "L_self_modules_patch_embed_modules_proj_parameters_weight_"),
    (
        [2401, 2],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [3, 384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [288, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_parameters_logit_scale_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [96, 384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [64, 49, 49],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_buffers_attn_mask_",
    ),
    (
        [2401, 2],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [3, 384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [288, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_parameters_logit_scale_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [96, 384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_weight_",
    ),
    (
        [2401, 2],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [6],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [6, 384],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [6],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_parameters_logit_scale_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [16, 49, 49],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_buffers_attn_mask_",
    ),
    (
        [2401, 2],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [6],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [6, 384],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [6],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_parameters_logit_scale_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [192, 384],
        "L_self_modules_stages_modules_1_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [2401, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [12, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [4, 49, 49],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_buffers_attn_mask_",
    ),
    (
        [2401, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [12, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [2401, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [12, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [4, 49, 49],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_buffers_attn_mask_",
    ),
    (
        [2401, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [12, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [2401, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [12, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [4, 49, 49],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_buffers_attn_mask_",
    ),
    (
        [2401, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [12, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [384, 768],
        "L_self_modules_stages_modules_2_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [2401, 2],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [24, 384],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_parameters_logit_scale_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [2401, 2],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_buffers_relative_coordinates_log_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 2],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [24, 384],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_meta_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_parameters_logit_scale_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_",
    ),
    ([1, 3, S0, S0], "L_x_"),
]
