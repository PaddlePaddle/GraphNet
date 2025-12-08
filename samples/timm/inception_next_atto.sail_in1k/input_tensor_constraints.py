from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([960], "L_self_modules_head_modules_fc1_parameters_bias_"),
    ([960, 320], "L_self_modules_head_modules_fc1_parameters_weight_"),
    ([1000], "L_self_modules_head_modules_fc2_parameters_bias_"),
    ([1000, 960], "L_self_modules_head_modules_fc2_parameters_weight_"),
    ([960], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([960], "L_self_modules_head_modules_norm_parameters_weight_"),
    (
        [160],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [160, 40, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [40, 160, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_buffers_running_var_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [10, 1, 9, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [10, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [10, 1, 1, 9],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [160, 40, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [40, 160, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_buffers_running_var_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [10, 1, 9, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [10, 1, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [10, 1, 1, 9],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [320, 80, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [80, 320, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_buffers_running_var_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [20, 1, 9, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [20, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [20, 1, 1, 9],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [320, 80, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [80, 320, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_buffers_running_var_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [20, 1, 9, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [20, 1, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [20, 1, 1, 9],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_mean_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_buffers_running_var_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_1_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [80, 40, 2, 2],
        "L_self_modules_stages_modules_1_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [640, 160, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [40, 1, 9, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [40, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [40, 1, 1, 9],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [640, 160, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [40, 1, 9, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [40, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [40, 1, 1, 9],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [640, 160, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [40, 1, 9, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [40, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [40, 1, 1, 9],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_parameters_gamma_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [640, 160, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [40, 1, 9, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [40, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [40, 1, 1, 9],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_parameters_gamma_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [640, 160, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [40, 1, 9, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [40, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [40, 1, 1, 9],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_parameters_gamma_",
    ),
    (
        [640],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [640, 160, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [40, 1, 9, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [40, 1, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [40, 1, 1, 9],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_parameters_gamma_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_buffers_running_var_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_2_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [160, 80, 2, 2],
        "L_self_modules_stages_modules_2_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [960, 320, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 960, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_buffers_running_var_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [80, 1, 9, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [80, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [80, 1, 1, 9],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_parameters_gamma_",
    ),
    (
        [960],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [960, 320, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [320, 960, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_buffers_running_var_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_bias_",
    ),
    (
        [80, 1, 9, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_h_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_bias_",
    ),
    (
        [80, 1, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_hw_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_bias_",
    ),
    (
        [80, 1, 1, 9],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_token_mixer_modules_dwconv_w_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_parameters_gamma_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_stages_modules_3_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [320, 160, 2, 2],
        "L_self_modules_stages_modules_3_modules_downsample_modules_1_parameters_weight_",
    ),
    ([40], "L_self_modules_stem_modules_0_parameters_bias_"),
    ([40, 3, 4, 4], "L_self_modules_stem_modules_0_parameters_weight_"),
    ([40], "L_self_modules_stem_modules_1_buffers_running_mean_"),
    ([40], "L_self_modules_stem_modules_1_buffers_running_var_"),
    ([40], "L_self_modules_stem_modules_1_parameters_bias_"),
    ([40], "L_self_modules_stem_modules_1_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
