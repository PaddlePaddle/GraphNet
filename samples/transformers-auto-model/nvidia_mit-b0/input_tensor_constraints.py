from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 3, 512, 512], "L_pixel_values_"),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [32, 32, 8, 8],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [128, 32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [32, 128],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [32, 32, 8, 8],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [128, 32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [32, 128],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [64, 64, 4, 4],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [256, 64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [64, 256],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [64, 64, 4, 4],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [256, 64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [64, 256],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [160, 160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [160, 160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [160, 160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [160, 160, 2, 2],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [160, 160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [640, 160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [160, 640],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [640, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [160, 160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [160, 160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [160, 160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [160, 160, 2, 2],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [160, 160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [640, 160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [160, 640],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [640, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [1024, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    ([32], "L_self_modules_encoder_modules_layer_norm_modules_0_parameters_bias_"),
    ([32], "L_self_modules_encoder_modules_layer_norm_modules_0_parameters_weight_"),
    ([64], "L_self_modules_encoder_modules_layer_norm_modules_1_parameters_bias_"),
    ([64], "L_self_modules_encoder_modules_layer_norm_modules_1_parameters_weight_"),
    ([160], "L_self_modules_encoder_modules_layer_norm_modules_2_parameters_bias_"),
    ([160], "L_self_modules_encoder_modules_layer_norm_modules_2_parameters_weight_"),
    ([256], "L_self_modules_encoder_modules_layer_norm_modules_3_parameters_bias_"),
    ([256], "L_self_modules_encoder_modules_layer_norm_modules_3_parameters_weight_"),
    (
        [32],
        "L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_bias_",
    ),
    (
        [32, 3, 7, 7],
        "L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_bias_",
    ),
    (
        [64, 32, 3, 3],
        "L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_bias_",
    ),
    (
        [160, 64, 3, 3],
        "L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_bias_",
    ),
    (
        [256, 160, 3, 3],
        "L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_weight_",
    ),
]
