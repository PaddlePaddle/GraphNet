dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 11], "L_attention_mask_"),
    ([1, 11], "L_input_ids_"),
    ([1536], "L_self_modules_embeddings_modules_LayerNorm_parameters_bias_"),
    ([1536], "L_self_modules_embeddings_modules_LayerNorm_parameters_weight_"),
    (
        [128100, 1536],
        "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    ([1536], "L_self_modules_encoder_modules_LayerNorm_parameters_bias_"),
    ([1536], "L_self_modules_encoder_modules_LayerNorm_parameters_weight_"),
    ([1536], "L_self_modules_encoder_modules_conv_modules_LayerNorm_parameters_bias_"),
    (
        [1536],
        "L_self_modules_encoder_modules_conv_modules_LayerNorm_parameters_weight_",
    ),
    ([1536], "L_self_modules_encoder_modules_conv_modules_conv_parameters_bias_"),
    (
        [1536, 1536, 3],
        "L_self_modules_encoder_modules_conv_modules_conv_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    ([512, 1536], "L_self_modules_encoder_modules_rel_embeddings_parameters_weight_"),
]
