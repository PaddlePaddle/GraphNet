from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 3, S1, S1], "L_pixel_values_"),
    ([], "L_self_modules_conv_1x1_modules_activation_max_val"),
    ([], "L_self_modules_conv_1x1_modules_activation_min_val"),
    (
        [1280, 320, 1, 1],
        "L_self_modules_conv_1x1_modules_convolution_parameters_weight_",
    ),
    ([1280], "L_self_modules_conv_1x1_modules_normalization_buffers_running_mean_"),
    ([1280], "L_self_modules_conv_1x1_modules_normalization_buffers_running_var_"),
    ([], "L_self_modules_conv_1x1_modules_normalization_eps"),
    ([], "L_self_modules_conv_1x1_modules_normalization_momentum"),
    ([1280], "L_self_modules_conv_1x1_modules_normalization_parameters_bias_"),
    ([1280], "L_self_modules_conv_1x1_modules_normalization_parameters_weight_"),
    ([], "L_self_modules_conv_stem_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_conv_stem_modules_conv_3x3_modules_activation_min_val"),
    (
        [32, 1, 3, 3],
        "L_self_modules_conv_stem_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_eps"),
    ([], "L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_momentum"),
    (
        [32],
        "L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_conv_stem_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_conv_stem_modules_first_conv_modules_activation_max_val"),
    ([], "L_self_modules_conv_stem_modules_first_conv_modules_activation_min_val"),
    (
        [32, 3, 3, 3],
        "L_self_modules_conv_stem_modules_first_conv_modules_convolution_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_conv_stem_modules_first_conv_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_conv_stem_modules_first_conv_modules_normalization_eps"),
    ([], "L_self_modules_conv_stem_modules_first_conv_modules_normalization_momentum"),
    (
        [32],
        "L_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_conv_stem_modules_first_conv_modules_normalization_parameters_weight_",
    ),
    (
        [16, 32, 1, 1],
        "L_self_modules_conv_stem_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_eps"),
    ([], "L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_momentum"),
    (
        [16],
        "L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_conv_stem_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_0_modules_conv_3x3_modules_activation_min_val"),
    (
        [96, 1, 3, 3],
        "L_self_modules_layer_modules_0_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [96],
        "L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_0_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_0_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [96, 16, 1, 1],
        "L_self_modules_layer_modules_0_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [96],
        "L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_0_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [24, 96, 1, 1],
        "L_self_modules_layer_modules_0_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [24],
        "L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [24],
        "L_self_modules_layer_modules_0_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_10_modules_conv_3x3_modules_activation_min_val"),
    (
        [576, 1, 3, 3],
        "L_self_modules_layer_modules_10_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [576],
        "L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_10_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_10_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [576, 96, 1, 1],
        "L_self_modules_layer_modules_10_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [576],
        "L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_10_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [96, 576, 1, 1],
        "L_self_modules_layer_modules_10_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [96],
        "L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_10_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_11_modules_conv_3x3_modules_activation_min_val"),
    (
        [576, 1, 3, 3],
        "L_self_modules_layer_modules_11_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [576],
        "L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_11_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_11_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [576, 96, 1, 1],
        "L_self_modules_layer_modules_11_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [576],
        "L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_11_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [96, 576, 1, 1],
        "L_self_modules_layer_modules_11_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [96],
        "L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_11_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_12_modules_conv_3x3_modules_activation_min_val"),
    (
        [576, 1, 3, 3],
        "L_self_modules_layer_modules_12_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [576],
        "L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_12_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_12_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [576, 96, 1, 1],
        "L_self_modules_layer_modules_12_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [576],
        "L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_layer_modules_12_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [160, 576, 1, 1],
        "L_self_modules_layer_modules_12_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [160],
        "L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_layer_modules_12_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_13_modules_conv_3x3_modules_activation_min_val"),
    (
        [960, 1, 3, 3],
        "L_self_modules_layer_modules_13_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [960],
        "L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_13_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_13_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_layer_modules_13_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [960],
        "L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_13_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_layer_modules_13_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [160],
        "L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_layer_modules_13_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_14_modules_conv_3x3_modules_activation_min_val"),
    (
        [960, 1, 3, 3],
        "L_self_modules_layer_modules_14_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [960],
        "L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_14_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_14_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_layer_modules_14_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [960],
        "L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_14_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_layer_modules_14_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [160],
        "L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_layer_modules_14_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_15_modules_conv_3x3_modules_activation_min_val"),
    (
        [960, 1, 3, 3],
        "L_self_modules_layer_modules_15_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [960],
        "L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_15_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_15_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_layer_modules_15_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [960],
        "L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [960],
        "L_self_modules_layer_modules_15_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [320, 960, 1, 1],
        "L_self_modules_layer_modules_15_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    (
        [],
        "L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_eps",
    ),
    (
        [],
        "L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [320],
        "L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_layer_modules_15_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_1_modules_conv_3x3_modules_activation_min_val"),
    (
        [144, 1, 3, 3],
        "L_self_modules_layer_modules_1_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [144],
        "L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_1_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_1_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [144, 24, 1, 1],
        "L_self_modules_layer_modules_1_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [144],
        "L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_1_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [24, 144, 1, 1],
        "L_self_modules_layer_modules_1_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [24],
        "L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [24],
        "L_self_modules_layer_modules_1_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_2_modules_conv_3x3_modules_activation_min_val"),
    (
        [144, 1, 3, 3],
        "L_self_modules_layer_modules_2_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [144],
        "L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_2_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_2_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [144, 24, 1, 1],
        "L_self_modules_layer_modules_2_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [144],
        "L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_layer_modules_2_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [32, 144, 1, 1],
        "L_self_modules_layer_modules_2_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [32],
        "L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_layer_modules_2_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_3_modules_conv_3x3_modules_activation_min_val"),
    (
        [192, 1, 3, 3],
        "L_self_modules_layer_modules_3_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [192],
        "L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_3_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_3_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [192, 32, 1, 1],
        "L_self_modules_layer_modules_3_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [192],
        "L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_3_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [32, 192, 1, 1],
        "L_self_modules_layer_modules_3_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [32],
        "L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_layer_modules_3_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_4_modules_conv_3x3_modules_activation_min_val"),
    (
        [192, 1, 3, 3],
        "L_self_modules_layer_modules_4_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [192],
        "L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_4_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_4_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [192, 32, 1, 1],
        "L_self_modules_layer_modules_4_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [192],
        "L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_4_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [32, 192, 1, 1],
        "L_self_modules_layer_modules_4_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [32],
        "L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_layer_modules_4_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_5_modules_conv_3x3_modules_activation_min_val"),
    (
        [192, 1, 3, 3],
        "L_self_modules_layer_modules_5_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [192],
        "L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_5_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_5_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [192, 32, 1, 1],
        "L_self_modules_layer_modules_5_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [192],
        "L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_layer_modules_5_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [64, 192, 1, 1],
        "L_self_modules_layer_modules_5_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [64],
        "L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_5_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_6_modules_conv_3x3_modules_activation_min_val"),
    (
        [384, 1, 3, 3],
        "L_self_modules_layer_modules_6_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [384],
        "L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_6_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_6_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_layer_modules_6_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [384],
        "L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_6_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [64, 384, 1, 1],
        "L_self_modules_layer_modules_6_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [64],
        "L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_6_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_7_modules_conv_3x3_modules_activation_min_val"),
    (
        [384, 1, 3, 3],
        "L_self_modules_layer_modules_7_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [384],
        "L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_7_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_7_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_layer_modules_7_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [384],
        "L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_7_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [64, 384, 1, 1],
        "L_self_modules_layer_modules_7_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [64],
        "L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_7_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_8_modules_conv_3x3_modules_activation_min_val"),
    (
        [384, 1, 3, 3],
        "L_self_modules_layer_modules_8_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [384],
        "L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_8_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_8_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_layer_modules_8_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [384],
        "L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_8_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [64, 384, 1, 1],
        "L_self_modules_layer_modules_8_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [64],
        "L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_layer_modules_8_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "L_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_max_val"),
    ([], "L_self_modules_layer_modules_9_modules_conv_3x3_modules_activation_min_val"),
    (
        [384, 1, 3, 3],
        "L_self_modules_layer_modules_9_modules_conv_3x3_modules_convolution_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_momentum",
    ),
    (
        [384],
        "L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_9_modules_conv_3x3_modules_normalization_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_max_val",
    ),
    (
        [],
        "L_self_modules_layer_modules_9_modules_expand_1x1_modules_activation_min_val",
    ),
    (
        [384, 64, 1, 1],
        "L_self_modules_layer_modules_9_modules_expand_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_momentum",
    ),
    (
        [384],
        "L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_layer_modules_9_modules_expand_1x1_modules_normalization_parameters_weight_",
    ),
    (
        [96, 384, 1, 1],
        "L_self_modules_layer_modules_9_modules_reduce_1x1_modules_convolution_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_buffers_running_var_",
    ),
    ([], "L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_eps"),
    (
        [],
        "L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_momentum",
    ),
    (
        [96],
        "L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_layer_modules_9_modules_reduce_1x1_modules_normalization_parameters_weight_",
    ),
    ([], "s99"),
]
