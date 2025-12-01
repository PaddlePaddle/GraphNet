dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, 448, 448], "L_pixel_values_"),
    ([], "L_self_modules_embedder_modules_convolution_eps"),
    ([64, 3, 7, 7], "L_self_modules_embedder_modules_convolution_parameters_weight_"),
    ([], "L_self_modules_embedder_modules_pad_value"),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_eps",
    ),
    (
        [8, 64, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_eps",
    ),
    (
        [8, 8, 3, 3],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_eps",
    ),
    (
        [8, 8, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_eps",
    ),
    (
        [8, 64, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_eps",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_eps",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_eps",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_norm3_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_eps",
    ),
    (
        [8, 8, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_eps",
    ),
    (
        [8, 8, 3, 3],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_eps",
    ),
    (
        [16, 8, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_eps",
    ),
    (
        [16, 8, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_eps",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_eps",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_eps",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_norm3_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_eps",
    ),
    (
        [8, 16, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_eps",
    ),
    (
        [8, 8, 3, 3],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_eps",
    ),
    (
        [32, 8, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_eps",
    ),
    (
        [32, 16, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_eps",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_eps",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_eps",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_norm3_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_eps",
    ),
    (
        [8, 32, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_eps",
    ),
    (
        [8, 8, 3, 3],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_eps",
    ),
    (
        [32, 8, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_eps",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_eps",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_norm3_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_eps",
    ),
    (
        [16, 32, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_eps",
    ),
    (
        [16, 16, 3, 3],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_eps",
    ),
    (
        [64, 16, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_eps",
    ),
    (
        [64, 32, 1, 1],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_eps",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_eps",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_norm3_parameters_weight_",
    ),
    ([], "L_self_modules_norm_eps"),
    ([64], "L_self_modules_norm_parameters_bias_"),
    ([64], "L_self_modules_norm_parameters_weight_"),
    ([], "s99"),
]
