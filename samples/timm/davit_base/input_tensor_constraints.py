from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 1024], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([1024], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([1024], "L_self_modules_head_modules_norm_parameters_weight_"),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [256, 128, 2, 2],
        "L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [512, 256, 2, 2],
        "L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe1_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_cpe2_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [1024, 512, 2, 2],
        "L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_",
    ),
    ([128], "L_self_modules_stem_modules_conv_parameters_bias_"),
    ([128, 3, 7, 7], "L_self_modules_stem_modules_conv_parameters_weight_"),
    ([128], "L_self_modules_stem_modules_norm_parameters_bias_"),
    ([128], "L_self_modules_stem_modules_norm_parameters_weight_"),
    ([S0, 3, 224, 224], "L_x_"),
]
