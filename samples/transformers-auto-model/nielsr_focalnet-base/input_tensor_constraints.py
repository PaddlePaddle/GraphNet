from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 3, S1, S1], "L_pixel_values_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1024], "L_self_modules_classifier_parameters_weight_"),
    ([128], "L_self_modules_focalnet_modules_embeddings_modules_norm_parameters_bias_"),
    (
        [128],
        "L_self_modules_focalnet_modules_embeddings_modules_norm_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_",
    ),
    (
        [128, 3, 4, 4],
        "L_self_modules_focalnet_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_downsample_modules_projection_parameters_bias_",
    ),
    (
        [256, 128, 2, 2],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_downsample_modules_projection_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [128, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [259],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [259, 128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [128, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [259],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [259, 128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_downsample_modules_projection_parameters_bias_",
    ),
    (
        [512, 256, 2, 2],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_downsample_modules_projection_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [515],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [515, 256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [515],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [515, 256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_downsample_modules_projection_parameters_bias_",
    ),
    (
        [1024, 512, 2, 2],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_downsample_modules_projection_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_10_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_11_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_12_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_13_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_14_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_15_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_16_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_17_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_6_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_7_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_8_modules_norm2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [512, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [1027],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [1027, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_2_modules_layers_modules_9_modules_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [2051],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [2051, 1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_modulation_modules_focal_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024, 1, 5, 5],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_modulation_modules_focal_layers_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_modulation_modules_projection_context_parameters_bias_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_modulation_modules_projection_context_parameters_weight_",
    ),
    (
        [2051],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_modulation_modules_projection_in_parameters_bias_",
    ),
    (
        [2051, 1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_modulation_modules_projection_in_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_modulation_modules_projection_out_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_modulation_modules_projection_out_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_focalnet_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_norm2_parameters_weight_",
    ),
    ([1024], "L_self_modules_focalnet_modules_layernorm_parameters_bias_"),
    ([1024], "L_self_modules_focalnet_modules_layernorm_parameters_weight_"),
]
