from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 1024], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [512],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [260],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [260, 128, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [128, 1, 5, 5],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [260],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [260, 128, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [128, 1, 5, 5],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_layers_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [516],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [516, 256, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256, 1, 5, 5],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [256, 1, 7, 7],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [516],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [516, 256, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256, 1, 5, 5],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [256, 1, 7, 7],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [256, 128, 2, 2],
        "L_self_modules_layers_modules_1_modules_downsample_modules_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_10_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_11_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_12_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_13_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_14_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_15_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_16_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_17_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_6_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_7_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_8_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1028],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [1028, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_blocks_modules_9_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [512, 256, 2, 2],
        "L_self_modules_layers_modules_2_modules_downsample_modules_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2052],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2052, 1024, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024, 1, 5, 5],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1024, 1, 7, 7],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2052],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_bias_",
    ),
    (
        [2052, 1024, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_f_parameters_weight_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024, 1, 5, 5],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1024, 1, 7, 7],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_focal_layers_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_bias_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_h_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_modulation_modules_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_bias_",
    ),
    (
        [1024, 512, 2, 2],
        "L_self_modules_layers_modules_3_modules_downsample_modules_proj_parameters_weight_",
    ),
    ([1024], "L_self_modules_norm_parameters_bias_"),
    ([1024], "L_self_modules_norm_parameters_weight_"),
    ([128], "L_self_modules_stem_modules_norm_parameters_bias_"),
    ([128], "L_self_modules_stem_modules_norm_parameters_weight_"),
    ([128], "L_self_modules_stem_modules_proj_parameters_bias_"),
    ([128, 3, 4, 4], "L_self_modules_stem_modules_proj_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
