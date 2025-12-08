from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([21842], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([21842, 2816], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls1_parameters_gamma_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls2_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1408, 352, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [352, 1408, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [708],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [708, 352, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [352, 1, 3, 3],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [352, 1, 5, 5],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [352, 1, 7, 7],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [352, 352, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [352, 352, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_bias_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_bias_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls1_parameters_gamma_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls2_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1408, 352, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [352, 1408, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [708],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [708, 352, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [352, 1, 3, 3],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [352, 1, 5, 5],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [352, 1, 7, 7],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [352, 352, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [352, 352, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_bias_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_bias_",
    ),
    (
        [352],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls1_parameters_gamma_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls2_parameters_gamma_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2816, 704, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [704, 2816, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1412],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1412, 704, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [704, 1, 3, 3],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [704, 1, 5, 5],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [704, 1, 7, 7],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [704, 704, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [704, 704, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_bias_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_bias_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls1_parameters_gamma_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls2_parameters_gamma_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2816, 704, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [704, 2816, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1412],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1412, 704, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [704, 1, 3, 3],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [704, 1, 5, 5],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [704, 1, 7, 7],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [704, 704, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [704, 704, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_bias_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_bias_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [704],
        "L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [704, 352, 3, 3],
        "L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls1_parameters_gamma_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls2_parameters_gamma_",
    ),
    (
        [5632],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [5632, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1408, 5632, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2820],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2820, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1408, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1408, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1408, 1408, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [1408, 704, 3, 3],
        "L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls1_parameters_gamma_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls2_parameters_gamma_",
    ),
    (
        [11264],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [11264, 2816, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2816, 11264, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [5636],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [5636, 2816, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [2816, 1, 3, 3],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [2816, 1, 5, 5],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [2816, 1, 7, 7],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [2816, 2816, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [2816, 2816, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_bias_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_bias_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls1_parameters_gamma_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls2_parameters_gamma_",
    ),
    (
        [11264],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [11264, 2816, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2816, 11264, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [5636],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [5636, 2816, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [2816, 1, 3, 3],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [2816, 1, 5, 5],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [2816, 1, 7, 7],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [2816, 2816, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_norm_parameters_bias_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_norm_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [2816, 2816, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_bias_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_bias_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [2816],
        "L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [2816, 1408, 3, 3],
        "L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_",
    ),
    ([2816], "L_self_modules_norm_parameters_bias_"),
    ([2816], "L_self_modules_norm_parameters_weight_"),
    ([352], "L_self_modules_stem_modules_norm_parameters_bias_"),
    ([352], "L_self_modules_stem_modules_norm_parameters_weight_"),
    ([352], "L_self_modules_stem_modules_proj_parameters_bias_"),
    ([352, 3, 7, 7], "L_self_modules_stem_modules_proj_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
