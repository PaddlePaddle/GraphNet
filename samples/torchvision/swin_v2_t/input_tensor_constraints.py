from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([96], "L_self_modules_features_modules_0_modules_0_parameters_bias_"),
    ([96, 3, 4, 4], "L_self_modules_features_modules_0_modules_0_parameters_weight_"),
    ([96], "L_self_modules_features_modules_0_modules_2_parameters_bias_"),
    ([96], "L_self_modules_features_modules_0_modules_2_parameters_weight_"),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_1_modules_0_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [3, 512],
        "L_self_modules_features_modules_1_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_features_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [288, 96],
        "L_self_modules_features_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3, 1, 1],
        "L_self_modules_features_modules_1_modules_0_modules_attn_parameters_logit_scale_",
    ),
    (
        [384],
        "L_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [384, 96],
        "L_self_modules_features_modules_1_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [96, 384],
        "L_self_modules_features_modules_1_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_1_modules_1_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [3, 512],
        "L_self_modules_features_modules_1_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_features_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [288, 96],
        "L_self_modules_features_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3, 1, 1],
        "L_self_modules_features_modules_1_modules_1_modules_attn_parameters_logit_scale_",
    ),
    (
        [384],
        "L_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [384, 96],
        "L_self_modules_features_modules_1_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [96, 384],
        "L_self_modules_features_modules_1_modules_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_features_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    ([192], "L_self_modules_features_modules_2_modules_norm_parameters_bias_"),
    ([192], "L_self_modules_features_modules_2_modules_norm_parameters_weight_"),
    (
        [192, 384],
        "L_self_modules_features_modules_2_modules_reduction_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_3_modules_0_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [6, 512],
        "L_self_modules_features_modules_3_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_features_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_features_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [6, 1, 1],
        "L_self_modules_features_modules_3_modules_0_modules_attn_parameters_logit_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_features_modules_3_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_features_modules_3_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_3_modules_1_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [6, 512],
        "L_self_modules_features_modules_3_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_features_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_features_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [6, 1, 1],
        "L_self_modules_features_modules_3_modules_1_modules_attn_parameters_logit_scale_",
    ),
    (
        [768],
        "L_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_features_modules_3_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_features_modules_3_modules_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_features_modules_3_modules_1_modules_norm2_parameters_weight_",
    ),
    ([384], "L_self_modules_features_modules_4_modules_norm_parameters_bias_"),
    ([384], "L_self_modules_features_modules_4_modules_norm_parameters_weight_"),
    (
        [384, 768],
        "L_self_modules_features_modules_4_modules_reduction_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_5_modules_0_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_features_modules_5_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_features_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_features_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_features_modules_5_modules_0_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_features_modules_5_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_features_modules_5_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_5_modules_1_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_features_modules_5_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_features_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_features_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_features_modules_5_modules_1_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_features_modules_5_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_features_modules_5_modules_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_5_modules_2_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_features_modules_5_modules_2_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_features_modules_5_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_features_modules_5_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_features_modules_5_modules_2_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_features_modules_5_modules_2_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_features_modules_5_modules_2_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_5_modules_3_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_features_modules_5_modules_3_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_features_modules_5_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_features_modules_5_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_features_modules_5_modules_3_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_features_modules_5_modules_3_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_features_modules_5_modules_3_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_5_modules_4_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_features_modules_5_modules_4_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_features_modules_5_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_features_modules_5_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_features_modules_5_modules_4_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_features_modules_5_modules_4_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_features_modules_5_modules_4_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_5_modules_5_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_features_modules_5_modules_5_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_features_modules_5_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1152, 384],
        "L_self_modules_features_modules_5_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_features_modules_5_modules_5_modules_attn_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_features_modules_5_modules_5_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_features_modules_5_modules_5_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_features_modules_5_modules_5_modules_norm2_parameters_weight_",
    ),
    ([768], "L_self_modules_features_modules_6_modules_norm_parameters_bias_"),
    ([768], "L_self_modules_features_modules_6_modules_norm_parameters_weight_"),
    (
        [768, 1536],
        "L_self_modules_features_modules_6_modules_reduction_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_7_modules_0_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [24, 512],
        "L_self_modules_features_modules_7_modules_0_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_features_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_features_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [24, 1, 1],
        "L_self_modules_features_modules_7_modules_0_modules_attn_parameters_logit_scale_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_7_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_7_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_coords_table_",
    ),
    (
        [4096],
        "L_self_modules_features_modules_7_modules_1_modules_attn_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_0_parameters_weight_",
    ),
    (
        [24, 512],
        "L_self_modules_features_modules_7_modules_1_modules_attn_modules_cpb_mlp_modules_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_features_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_features_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [24, 1, 1],
        "L_self_modules_features_modules_7_modules_1_modules_attn_parameters_logit_scale_",
    ),
    (
        [3072],
        "L_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_features_modules_7_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_features_modules_7_modules_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_features_modules_7_modules_1_modules_norm2_parameters_weight_",
    ),
    ([1000], "L_self_modules_head_parameters_bias_"),
    ([1000, 768], "L_self_modules_head_parameters_weight_"),
    ([768], "L_self_modules_norm_parameters_bias_"),
    ([768], "L_self_modules_norm_parameters_weight_"),
    ([1, 3, S0, S0], "L_x_"),
]
