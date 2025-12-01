from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([2048], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([2048], "L_self_modules_head_modules_norm_parameters_weight_"),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [1024, 512, 3, 3],
        "L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [2048, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [2048, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_norm1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_norm1_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_norm2_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_channel_block_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [2048, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [2048, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_norm1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_norm1_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_norm2_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_block_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [2048, 1024, 3, 3],
        "L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_",
    ),
    ([256], "L_self_modules_stem_modules_conv_parameters_bias_"),
    ([256, 3, 7, 7], "L_self_modules_stem_modules_conv_parameters_weight_"),
    ([256], "L_self_modules_stem_modules_norm_parameters_bias_"),
    ([256], "L_self_modules_stem_modules_norm_parameters_weight_"),
    ([S0, 3, 768, 768], "L_x_"),
]
