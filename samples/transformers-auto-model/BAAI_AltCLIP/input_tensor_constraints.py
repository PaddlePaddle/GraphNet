from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([2, 7], "L_attention_mask_"),
    ([2, 7], "L_input_ids_"),
    ([1, 3, S0, S0], "L_pixel_values_"),
    ([1024], "L_self_modules_text_model_modules_pre_LN_parameters_bias_"),
    ([1024], "L_self_modules_text_model_modules_pre_LN_parameters_weight_"),
    (
        [1, 514],
        "L_self_modules_text_model_modules_roberta_modules_embeddings_buffers_token_type_ids_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_embeddings_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_embeddings_modules_LayerNorm_parameters_weight_",
    ),
    (
        [514, 1024],
        "L_self_modules_text_model_modules_roberta_modules_embeddings_modules_position_embeddings_parameters_weight_",
    ),
    (
        [1, 1024],
        "L_self_modules_text_model_modules_roberta_modules_embeddings_modules_token_type_embeddings_parameters_weight_",
    ),
    (
        [250002, 1024],
        "L_self_modules_text_model_modules_roberta_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_text_model_modules_roberta_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    ([768], "L_self_modules_text_model_modules_transformation_parameters_bias_"),
    (
        [768, 1024],
        "L_self_modules_text_model_modules_transformation_parameters_weight_",
    ),
    ([768, 768], "L_self_modules_text_projection_parameters_weight_"),
    ([1, 257], "L_self_modules_vision_model_modules_embeddings_buffers_position_ids_"),
    (
        [1024, 3, 14, 14],
        "L_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_",
    ),
    (
        [257, 1024],
        "L_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_embeddings_parameters_class_embedding_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([1024], "L_self_modules_vision_model_modules_post_layernorm_parameters_bias_"),
    ([1024], "L_self_modules_vision_model_modules_post_layernorm_parameters_weight_"),
    ([1024], "L_self_modules_vision_model_modules_pre_layrnorm_parameters_bias_"),
    ([1024], "L_self_modules_vision_model_modules_pre_layrnorm_parameters_weight_"),
    ([768, 1024], "L_self_modules_visual_projection_parameters_weight_"),
    ([], "L_self_parameters_logit_scale_"),
]
