from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1280], "L_self_modules_classifier_parameters_weight_"),
    ([1280, 1152, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([144], "L_self_modules_patch_embed_modules_norm_buffers_running_mean_"),
    ([144], "L_self_modules_patch_embed_modules_norm_buffers_running_var_"),
    ([144], "L_self_modules_patch_embed_modules_norm_parameters_bias_"),
    ([144], "L_self_modules_patch_embed_modules_norm_parameters_weight_"),
    ([144, 3, 4, 4], "L_self_modules_patch_embed_modules_proj_parameters_weight_"),
    (
        [288, 144, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [144, 288, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [36, 36, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [288, 144, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [144, 288, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [36, 36, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [288, 144, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [144, 288, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [36, 36, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [576, 288, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [288, 576, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [72, 72, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [576, 288, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [288, 576, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [72, 72, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [576, 288, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [288, 576, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [72, 72, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [576, 288, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [288, 576, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [72, 72, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [288],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [288, 144, 2, 2],
        "L_self_modules_stages_modules_1_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152, 576, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [576, 1152, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [144, 144, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_buffers_running_var_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [576, 288, 2, 2],
        "L_self_modules_stages_modules_2_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [2304, 1152, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [1152, 2304, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [288, 288, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [2304, 1152, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [1152, 2304, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [288, 288, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [2304, 1152, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_1_buffers_running_mean_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_1_buffers_running_var_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_1_parameters_bias_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_1_parameters_weight_",
    ),
    (
        [1152, 2304, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [288, 288, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_spatial_mixing_modules_partial_conv3_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_buffers_running_var_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [1152, 576, 2, 2],
        "L_self_modules_stages_modules_3_modules_downsample_modules_reduction_parameters_weight_",
    ),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
