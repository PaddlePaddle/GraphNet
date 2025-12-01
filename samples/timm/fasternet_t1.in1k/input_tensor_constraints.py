from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1280], "L_self_modules_classifier_parameters_weight_"),
    ([1280, 512, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([64], "L_self_modules_patch_embed_modules_norm_buffers_running_mean_"),
    ([64], "L_self_modules_patch_embed_modules_norm_buffers_running_var_"),
    ([64], "L_self_modules_patch_embed_modules_norm_parameters_bias_"),
    ([64], "L_self_modules_patch_embed_modules_norm_parameters_weight_"),
    ([64, 3, 4, 4], "L_self_modules_patch_embed_modules_proj_parameters_weight_"),
    (
        [128, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [64, 128, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [16, 16, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [256, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [256, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [32, 32, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_buffers_running_var_",
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
        [128, 64, 2, 2],
        "L_self_modules_stages_modules_1_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [256, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [256, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [256, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [256, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [256, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [256, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [256, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [256, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_buffers_running_var_",
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
        [256, 128, 2, 2],
        "L_self_modules_stages_modules_2_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [1024, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [512, 1024, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1024, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [512, 1024, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [512, 256, 2, 2],
        "L_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_",
    ),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
