dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 15], "L_attention_mask_"),
    ([1, 15], "L_input_ids_"),
    ([1, 514], "L_self_modules_embeddings_buffers_token_type_ids_"),
    ([384], "L_self_modules_embeddings_modules_LayerNorm_parameters_bias_"),
    ([384], "L_self_modules_embeddings_modules_LayerNorm_parameters_weight_"),
    (
        [514, 384],
        "L_self_modules_embeddings_modules_position_embeddings_parameters_weight_",
    ),
    (
        [1, 384],
        "L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_",
    ),
    (
        [250002, 384],
        "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    ([384], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([384, 384], "L_self_modules_pooler_modules_dense_parameters_weight_"),
]
