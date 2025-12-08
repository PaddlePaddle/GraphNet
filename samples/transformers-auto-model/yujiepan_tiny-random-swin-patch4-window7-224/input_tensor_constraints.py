from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, S0, S0], "L_pixel_values_"),
    ([8], "L_self_modules_embeddings_modules_norm_parameters_bias_"),
    ([8], "L_self_modules_embeddings_modules_norm_parameters_weight_"),
    (
        [8],
        "L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_",
    ),
    (
        [8, 3, 4, 4],
        "L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [8, 8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [8, 8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [8, 8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [8, 8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 2],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [16, 8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [8, 16],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [16, 32],
        "L_self_modules_encoder_modules_layers_modules_0_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 4],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [32, 16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [16, 32],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [32, 64],
        "L_self_modules_encoder_modules_layers_modules_1_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 8],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [64, 32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 64],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [64, 128],
        "L_self_modules_encoder_modules_layers_modules_2_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 16],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [128, 64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [64, 128],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    ([64], "L_self_modules_layernorm_parameters_bias_"),
    ([64], "L_self_modules_layernorm_parameters_weight_"),
]
