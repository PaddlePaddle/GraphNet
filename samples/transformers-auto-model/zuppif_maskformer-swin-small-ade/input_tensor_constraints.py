dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, 512, 512], "L_pixel_values_"),
    (
        [256, 256, 3, 3],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_0_modules_block_layers_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_0_modules_block_layers_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_0_modules_block_layers_1_parameters_weight_",
    ),
    (
        [256, 384, 1, 1],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_0_modules_proj_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_0_modules_proj_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_0_modules_proj_modules_1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_1_modules_block_layers_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_1_modules_block_layers_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_1_modules_block_layers_1_parameters_weight_",
    ),
    (
        [256, 192, 1, 1],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_1_modules_proj_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_1_modules_proj_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_1_modules_proj_modules_1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_2_modules_block_layers_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_2_modules_block_layers_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_2_modules_block_layers_1_parameters_weight_",
    ),
    (
        [256, 96, 1, 1],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_2_modules_proj_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_2_modules_proj_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_layers_modules_2_modules_proj_modules_1_parameters_weight_",
    ),
    (
        [256, 768, 3, 3],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_stem_layers_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_stem_layers_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_fpn_modules_stem_layers_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_pixel_level_module_modules_decoder_modules_mask_projection_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_pixel_level_module_modules_decoder_modules_mask_projection_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_hidden_states_norms_modules_0_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_hidden_states_norms_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_hidden_states_norms_modules_1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_hidden_states_norms_modules_1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_hidden_states_norms_modules_2_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_hidden_states_norms_modules_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_hidden_states_norms_modules_3_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_hidden_states_norms_modules_3_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_embeddings_modules_norm_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_embeddings_modules_norm_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_",
    ),
    (
        [96, 3, 4, 4],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 3],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [384, 96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [96, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [96, 96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 3],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [384, 96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [96, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [192, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_0_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 6],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 6],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [384, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_1_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_12_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_13_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_14_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_15_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_16_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_17_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 12],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_blocks_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_2_modules_downsample_modules_reduction_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 24],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [49, 49],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [169, 24],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_attention_modules_self_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_encoder_modules_layers_modules_3_modules_blocks_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_layernorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_pixel_level_module_modules_encoder_modules_model_modules_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layernorm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_",
    ),
    (
        [256, 2048],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_transformer_module_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_transformer_module_modules_input_projection_parameters_bias_",
    ),
    (
        [256, 768, 1, 1],
        "L_self_modules_transformer_module_modules_input_projection_parameters_weight_",
    ),
    (
        [100, 256],
        "L_self_modules_transformer_module_modules_queries_embedder_parameters_weight_",
    ),
]
