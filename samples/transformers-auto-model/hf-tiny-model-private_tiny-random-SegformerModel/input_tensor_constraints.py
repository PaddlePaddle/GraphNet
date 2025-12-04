dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, 64, 64], "L_pixel_values_"),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [16, 16, 8, 8],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [64, 16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [16, 64],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [64, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_0_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [16, 16, 8, 8],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [64, 16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [16, 64],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [64, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_0_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [32, 32, 4, 4],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [128, 32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [32, 128],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_1_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [32, 32, 4, 4],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [128, 32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [32, 128],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [128, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_1_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [64, 64, 2, 2],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [256, 64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [64, 256],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_2_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_layer_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_bias_",
    ),
    (
        [64, 64, 2, 2],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_sr_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [256, 64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [64, 256],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [256, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_2_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_3_modules_0_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_layer_norm_2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_bias_",
    ),
    (
        [512, 1, 3, 3],
        "L_self_modules_encoder_modules_block_modules_3_modules_1_modules_mlp_modules_dwconv_modules_dwconv_parameters_weight_",
    ),
    ([16], "L_self_modules_encoder_modules_layer_norm_modules_0_parameters_bias_"),
    ([16], "L_self_modules_encoder_modules_layer_norm_modules_0_parameters_weight_"),
    ([32], "L_self_modules_encoder_modules_layer_norm_modules_1_parameters_bias_"),
    ([32], "L_self_modules_encoder_modules_layer_norm_modules_1_parameters_weight_"),
    ([64], "L_self_modules_encoder_modules_layer_norm_modules_2_parameters_bias_"),
    ([64], "L_self_modules_encoder_modules_layer_norm_modules_2_parameters_weight_"),
    ([128], "L_self_modules_encoder_modules_layer_norm_modules_3_parameters_bias_"),
    ([128], "L_self_modules_encoder_modules_layer_norm_modules_3_parameters_weight_"),
    (
        [16],
        "L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_bias_",
    ),
    (
        [16, 3, 7, 7],
        "L_self_modules_encoder_modules_patch_embeddings_modules_0_modules_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_bias_",
    ),
    (
        [32, 16, 3, 3],
        "L_self_modules_encoder_modules_patch_embeddings_modules_1_modules_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_bias_",
    ),
    (
        [64, 32, 3, 3],
        "L_self_modules_encoder_modules_patch_embeddings_modules_2_modules_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_layer_norm_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_bias_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_encoder_modules_patch_embeddings_modules_3_modules_proj_parameters_weight_",
    ),
]
