from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 13}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_kwargs_attention_mask_"),
    ([S0, S1], "L_kwargs_input_ids_"),
    ([33, 480], "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_"),
    ([480], "L_self_modules_encoder_modules_emb_layer_norm_after_parameters_bias_"),
    ([480], "L_self_modules_encoder_modules_emb_layer_norm_after_parameters_weight_"),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [12],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_rotary_embeddings_buffers_inv_freq_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [480, 480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1920, 480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [480, 1920],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    ([480], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([480, 480], "L_self_modules_pooler_modules_dense_parameters_weight_"),
]
