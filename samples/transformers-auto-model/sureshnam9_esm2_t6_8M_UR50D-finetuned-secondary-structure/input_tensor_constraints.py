dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 13], "L_kwargs_attention_mask_"),
    ([1, 13], "L_kwargs_input_ids_"),
    ([33, 320], "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_"),
    ([320], "L_self_modules_encoder_modules_emb_layer_norm_after_parameters_bias_"),
    ([320], "L_self_modules_encoder_modules_emb_layer_norm_after_parameters_weight_"),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [320, 320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1280, 320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [320, 1280],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    ([320], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([320, 320], "L_self_modules_pooler_modules_dense_parameters_weight_"),
]
