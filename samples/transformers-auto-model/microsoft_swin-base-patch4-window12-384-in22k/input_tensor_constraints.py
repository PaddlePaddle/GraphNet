from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 384}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, S0, S0], "L_pixel_values_"),
    ([128], "L_self_modules_embeddings_modules_norm_parameters_bias_"),
    ([128], "L_self_modules_embeddings_modules_norm_parameters_weight_"),
    (
        [128],
        "L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_",
    ),
    (
        [128, 3, 4, 4],
        "L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 4],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 4],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [512, 128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [128, 512],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [256, 512],
        "L_self_modules_encoder_modules_layers_modules_0_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 8],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 8],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_encoder_modules_layers_modules_1_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 16],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_layernorm_after_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_layernorm_after_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_layernorm_before_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_layernorm_before_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_encoder_modules_layers_modules_2_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 32],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [144, 144],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [529, 32],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_layernorm_parameters_bias_"),
    ([1024], "L_self_modules_layernorm_parameters_weight_"),
]
