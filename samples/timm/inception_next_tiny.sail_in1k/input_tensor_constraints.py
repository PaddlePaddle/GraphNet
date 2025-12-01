from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([2304], "L_self_modules_head_modules_fc1_parameters_bias_"),
    ([2304, 768], "L_self_modules_head_modules_fc1_parameters_weight_"),
    ([1000], "L_self_modules_head_modules_fc2_parameters_bias_"),
    ([1000, 2304], "L_self_modules_head_modules_fc2_parameters_weight_"),
    ([2304], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([2304], "L_self_modules_head_modules_norm_parameters_weight_"),
    (
        [384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [12, 1, 11, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [12, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [12, 1, 1, 11],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [12, 1, 11, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [12, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [12, 1, 1, 11],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [12, 1, 11, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [12, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [12, 1, 1, 11],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [24, 1, 11, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [24, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [24, 1, 1, 11],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [24, 1, 11, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [24, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [24, 1, 1, 11],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [24, 1, 11, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [24, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [24, 1, 1, 11],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_parameters_gamma_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [192, 96, 2, 2],
        "L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [48, 1, 11, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [48, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [48, 1, 1, 11],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [48, 1, 11, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [48, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [48, 1, 1, 11],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [48, 1, 11, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [48, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [48, 1, 1, 11],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [48, 1, 11, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [48, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [48, 1, 1, 11],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [48, 1, 11, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [48, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [48, 1, 1, 11],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [48, 1, 11, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [48, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [48, 1, 1, 11],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [48, 1, 11, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [48, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [48, 1, 1, 11],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [48, 1, 11, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [48, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [48, 1, 1, 11],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [48, 1, 11, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [48, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [48, 1, 1, 11],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_parameters_gamma_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [384, 192, 2, 2],
        "L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2304, 768, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 2304, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [96, 1, 11, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [96, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [96, 1, 1, 11],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2304, 768, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 2304, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [96, 1, 11, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [96, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [96, 1, 1, 11],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [2304],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2304, 768, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 2304, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [96, 1, 11, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [96, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [96, 1, 1, 11],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_parameters_gamma_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [768, 384, 2, 2],
        "L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_",
    ),
    ([96], "L_self_modules_stem_modules_0_parameters_bias_"),
    ([96, 3, 4, 4], "L_self_modules_stem_modules_0_parameters_weight_"),
    ([96], "L_self_modules_stem_modules_1_buffers_running_mean_"),
    ([96], "L_self_modules_stem_modules_1_buffers_running_var_"),
    ([96], "L_self_modules_stem_modules_1_parameters_bias_"),
    ([96], "L_self_modules_stem_modules_1_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
