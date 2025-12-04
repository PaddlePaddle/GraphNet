from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 256, S1: 192}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, S0, S1], "L_inputs_"),
    ([1024], "L_self_modules_backbone_modules_norm3_parameters_bias_"),
    ([1024], "L_self_modules_backbone_modules_norm3_parameters_weight_"),
    (
        [128],
        "L_self_modules_backbone_modules_patch_embed_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_patch_embed_modules_norm_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_",
    ),
    (
        [128, 3, 4, 4],
        "L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 4],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 4],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_0_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_0_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [256, 512],
        "L_self_modules_backbone_modules_stages_modules_0_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 8],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 8],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_backbone_modules_stages_modules_1_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_stages_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_backbone_modules_stages_modules_2_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 32],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_w_msa_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_w_msa_modules_qkv_parameters_weight_",
    ),
    (
        [169, 32],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_w_msa_parameters_relative_position_bias_table_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [1024, 256, 4, 4],
        "L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_",
    ),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_"),
    (
        [256, 256, 4, 4],
        "L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_",
    ),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_"),
    (
        [256, 256, 4, 4],
        "L_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_",
    ),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_"),
    ([17], "L_self_modules_head_modules_final_layer_parameters_bias_"),
    ([17, 256, 1, 1], "L_self_modules_head_modules_final_layer_parameters_weight_"),
]
