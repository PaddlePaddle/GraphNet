from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1024}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, S0, S0], "L_pixel_values_"),
    ([96], "L_self_modules_embeddings_modules_norm_parameters_bias_"),
    ([96], "L_self_modules_embeddings_modules_norm_parameters_weight_"),
    (
        [96],
        "L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_",
    ),
    (
        [96, 3, 4, 4],
        "L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [3, 512],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [96, 96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [384, 96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [96, 384],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [3, 512],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [96, 96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [384, 96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [96, 384],
        "L_self_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [192, 384],
        "L_self_modules_encoder_modules_layers_modules_0_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [6, 512],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [6, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [6, 512],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [192, 192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [6, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [384, 768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [12, 512],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [12, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_encoder_modules_layers_modules_2_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [24, 512],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [24, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1, 15, 15, 2],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_coords_table_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_bias_",
    ),
    (
        [512, 2],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_0_parameters_weight_",
    ),
    (
        [24, 512],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_continuous_position_bias_mlp_modules_2_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [24, 1, 1],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_parameters_logit_scale_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    ([768], "L_self_modules_layernorm_parameters_bias_"),
    ([768], "L_self_modules_layernorm_parameters_weight_"),
]
