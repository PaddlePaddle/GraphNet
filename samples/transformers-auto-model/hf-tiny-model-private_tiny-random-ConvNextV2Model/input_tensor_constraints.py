from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 3, S1, S1], "L_pixel_values_"),
    ([], "L_self_modules_embeddings_modules_layernorm_eps"),
    ([10], "L_self_modules_embeddings_modules_layernorm_parameters_bias_"),
    ([10], "L_self_modules_embeddings_modules_layernorm_parameters_weight_"),
    ([10], "L_self_modules_embeddings_modules_patch_embeddings_parameters_bias_"),
    (
        [10, 3, 4, 4],
        "L_self_modules_embeddings_modules_patch_embeddings_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_bias_",
    ),
    (
        [10, 1, 7, 7],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_dwconv_parameters_weight_",
    ),
    (
        [1, 1, 1, 40],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_grn_parameters_bias_",
    ),
    (
        [1, 1, 1, 40],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_grn_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_eps",
    ),
    (
        [10],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_bias_",
    ),
    (
        [10],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_layernorm_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_bias_",
    ),
    (
        [40, 10],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv1_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_bias_",
    ),
    (
        [10, 40],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_0_modules_pwconv2_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_bias_",
    ),
    (
        [10, 1, 7, 7],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_dwconv_parameters_weight_",
    ),
    (
        [1, 1, 1, 40],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_grn_parameters_bias_",
    ),
    (
        [1, 1, 1, 40],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_grn_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_eps",
    ),
    (
        [10],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_bias_",
    ),
    (
        [10],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_layernorm_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_bias_",
    ),
    (
        [40, 10],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv1_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_bias_",
    ),
    (
        [10, 40],
        "L_self_modules_encoder_modules_stages_modules_0_modules_layers_modules_1_modules_pwconv2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_eps",
    ),
    (
        [10],
        "L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_bias_",
    ),
    (
        [10],
        "L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_0_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_bias_",
    ),
    (
        [20, 10, 2, 2],
        "L_self_modules_encoder_modules_stages_modules_1_modules_downsampling_layer_modules_1_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_bias_",
    ),
    (
        [20, 1, 7, 7],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_dwconv_parameters_weight_",
    ),
    (
        [1, 1, 1, 80],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_grn_parameters_bias_",
    ),
    (
        [1, 1, 1, 80],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_grn_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_eps",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_layernorm_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_bias_",
    ),
    (
        [80, 20],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv1_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_bias_",
    ),
    (
        [20, 80],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_0_modules_pwconv2_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_bias_",
    ),
    (
        [20, 1, 7, 7],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_dwconv_parameters_weight_",
    ),
    (
        [1, 1, 1, 80],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_grn_parameters_bias_",
    ),
    (
        [1, 1, 1, 80],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_grn_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_eps",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_layernorm_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_bias_",
    ),
    (
        [80, 20],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv1_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_bias_",
    ),
    (
        [20, 80],
        "L_self_modules_encoder_modules_stages_modules_1_modules_layers_modules_1_modules_pwconv2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_eps",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_0_parameters_weight_",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_bias_",
    ),
    (
        [30, 20, 2, 2],
        "L_self_modules_encoder_modules_stages_modules_2_modules_downsampling_layer_modules_1_parameters_weight_",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_bias_",
    ),
    (
        [30, 1, 7, 7],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_dwconv_parameters_weight_",
    ),
    (
        [1, 1, 1, 120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_grn_parameters_bias_",
    ),
    (
        [1, 1, 1, 120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_grn_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_eps",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_bias_",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_layernorm_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_bias_",
    ),
    (
        [120, 30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv1_parameters_weight_",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_bias_",
    ),
    (
        [30, 120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_0_modules_pwconv2_parameters_weight_",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_bias_",
    ),
    (
        [30, 1, 7, 7],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_dwconv_parameters_weight_",
    ),
    (
        [1, 1, 1, 120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_grn_parameters_bias_",
    ),
    (
        [1, 1, 1, 120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_grn_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_eps",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_bias_",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_layernorm_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_bias_",
    ),
    (
        [120, 30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv1_parameters_weight_",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_bias_",
    ),
    (
        [30, 120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_1_modules_pwconv2_parameters_weight_",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_bias_",
    ),
    (
        [30, 1, 7, 7],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_dwconv_parameters_weight_",
    ),
    (
        [1, 1, 1, 120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_grn_parameters_bias_",
    ),
    (
        [1, 1, 1, 120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_grn_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_eps",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_bias_",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_layernorm_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_bias_",
    ),
    (
        [120, 30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv1_parameters_weight_",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_bias_",
    ),
    (
        [30, 120],
        "L_self_modules_encoder_modules_stages_modules_2_modules_layers_modules_2_modules_pwconv2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_eps",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_bias_",
    ),
    (
        [30],
        "L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_0_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_bias_",
    ),
    (
        [40, 30, 2, 2],
        "L_self_modules_encoder_modules_stages_modules_3_modules_downsampling_layer_modules_1_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_bias_",
    ),
    (
        [40, 1, 7, 7],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_dwconv_parameters_weight_",
    ),
    (
        [1, 1, 1, 160],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_grn_parameters_bias_",
    ),
    (
        [1, 1, 1, 160],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_grn_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_eps",
    ),
    (
        [40],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_bias_",
    ),
    (
        [40],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_layernorm_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_bias_",
    ),
    (
        [160, 40],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv1_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_bias_",
    ),
    (
        [40, 160],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_0_modules_pwconv2_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_bias_",
    ),
    (
        [40, 1, 7, 7],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_dwconv_parameters_weight_",
    ),
    (
        [1, 1, 1, 160],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_grn_parameters_bias_",
    ),
    (
        [1, 1, 1, 160],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_grn_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_eps",
    ),
    (
        [40],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_bias_",
    ),
    (
        [40],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_layernorm_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_bias_",
    ),
    (
        [160, 40],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv1_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_bias_",
    ),
    (
        [40, 160],
        "L_self_modules_encoder_modules_stages_modules_3_modules_layers_modules_1_modules_pwconv2_parameters_weight_",
    ),
    ([], "L_self_modules_layernorm_eps"),
    ([40], "L_self_modules_layernorm_parameters_bias_"),
    ([40], "L_self_modules_layernorm_parameters_weight_"),
    ([], "s99"),
]
