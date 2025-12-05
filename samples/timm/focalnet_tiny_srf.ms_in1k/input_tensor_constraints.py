from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 768], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [384],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [195],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [195, 96, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [96, 1, 3, 3],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [96, 1, 5, 5],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 96, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [195],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [195, 96, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [96, 1, 3, 3],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [96, 1, 5, 5],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [96, 96, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [387],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [387, 192, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [192, 1, 3, 3],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [387],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [387, 192, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [192, 1, 3, 3],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [192, 1, 5, 5],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [192, 192, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [192, 96, 2, 2],
        "L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [771],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [771, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [771],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [771, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [771],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [771, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [771],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [771, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [771],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [771, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [771],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [771, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [384, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [384, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [384, 384, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [384, 192, 2, 2],
        "L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1539],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1539, 768, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1539],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1539, 768, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [768, 1, 3, 3],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768, 1, 5, 5],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [768, 384, 2, 2],
        "L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_norm_parameters_bias_"),
    ([768], "L_self_modules_norm_parameters_weight_"),
    ([96], "L_self_modules_stem_modules_norm_parameters_bias_"),
    ([96], "L_self_modules_stem_modules_norm_parameters_weight_"),
    ([96], "L_self_modules_stem_modules_proj_parameters_bias_"),
    ([96, 3, 4, 4], "L_self_modules_stem_modules_proj_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
