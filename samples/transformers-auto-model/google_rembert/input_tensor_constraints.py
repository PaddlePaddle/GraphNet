from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 13], "L_attention_mask_"),
    ([S0, 13], "L_input_ids_"),
    ([1, 512], "L_self_modules_embeddings_buffers_position_ids_"),
    ([256], "L_self_modules_embeddings_modules_LayerNorm_parameters_bias_"),
    ([256], "L_self_modules_embeddings_modules_LayerNorm_parameters_weight_"),
    (
        [512, 256],
        "L_self_modules_embeddings_modules_position_embeddings_parameters_weight_",
    ),
    (
        [2, 256],
        "L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_",
    ),
    (
        [250300, 256],
        "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_bias_",
    ),
    (
        [1152, 256],
        "L_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [1152, 1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4608, 1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1152, 4608],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    ([1152], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([1152, 1152], "L_self_modules_pooler_modules_dense_parameters_weight_"),
    ([S0, 13], "L_token_type_ids_"),
]
