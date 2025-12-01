from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([21842], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([21842, 1536], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls1_parameters_gamma_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_ls2_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [389],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [389, 192, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [192, 1, 3, 3],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 9, 9],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_post_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_post_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls1_parameters_gamma_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_ls2_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [389],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [389, 192, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [192, 1, 3, 3],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 9, 9],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_post_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_post_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls1_parameters_gamma_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_ls2_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [773],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [773, 384, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 5, 5],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 7, 7],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 9, 9],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_post_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_post_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls1_parameters_gamma_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_ls2_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [773],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [773, 384, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 5, 5],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 7, 7],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 9, 9],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_post_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_post_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [384, 192, 3, 3],
        "L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls1_parameters_gamma_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_ls2_parameters_gamma_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1541],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1541, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 9, 9],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_post_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [768, 384, 3, 3],
        "L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls1_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_ls2_parameters_gamma_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [6144, 1536, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1536, 6144, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [3077],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [3077, 1536, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1536, 1, 3, 3],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1536, 1, 5, 5],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1536, 1, 7, 7],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1536, 1, 9, 9],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1536, 1536, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1536, 1536, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_post_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_post_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls1_parameters_gamma_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_ls2_parameters_gamma_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [6144, 1536, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1536, 6144, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [3077],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [3077, 1536, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1536, 1, 3, 3],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1536, 1, 5, 5],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1536, 1, 7, 7],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1536, 1, 9, 9],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_3_modules_0_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1536, 1536, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1536, 1536, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_post_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_post_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [1536, 768, 3, 3],
        "L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_",
    ),
    ([1536], "L_self_modules_norm_parameters_bias_"),
    ([1536], "L_self_modules_norm_parameters_weight_"),
    ([192], "L_self_modules_stem_modules_norm_parameters_bias_"),
    ([192], "L_self_modules_stem_modules_norm_parameters_weight_"),
    ([192], "L_self_modules_stem_modules_proj_parameters_bias_"),
    ([192, 3, 7, 7], "L_self_modules_stem_modules_proj_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
