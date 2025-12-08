from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 384}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([2, 7], "L_attention_mask_"),
    ([2, 7], "L_input_ids_"),
    ([1, 3, S0, S0], "L_pixel_values_"),
    ([1, 512], "L_self_modules_text_model_modules_embeddings_buffers_position_ids_"),
    (
        [768],
        "L_self_modules_text_model_modules_embeddings_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_embeddings_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512, 768],
        "L_self_modules_text_model_modules_embeddings_modules_position_embeddings_parameters_weight_",
    ),
    (
        [30524, 768],
        "L_self_modules_text_model_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    ([768], "L_self_modules_text_model_modules_pooler_modules_dense_parameters_bias_"),
    (
        [768, 768],
        "L_self_modules_text_model_modules_pooler_modules_dense_parameters_weight_",
    ),
    ([512, 768], "L_self_modules_text_projection_parameters_weight_"),
    (
        [768],
        "L_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_bias_",
    ),
    (
        [768, 3, 16, 16],
        "L_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_",
    ),
    (
        [1, 1, 768],
        "L_self_modules_vision_model_modules_embeddings_parameters_class_embedding_",
    ),
    (
        [1, 577, 768],
        "L_self_modules_vision_model_modules_embeddings_parameters_position_embedding_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_projection_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    ([768], "L_self_modules_vision_model_modules_post_layernorm_parameters_bias_"),
    ([768], "L_self_modules_vision_model_modules_post_layernorm_parameters_weight_"),
    ([512, 768], "L_self_modules_visual_projection_parameters_weight_"),
    ([], "L_self_parameters_logit_scale_"),
]
