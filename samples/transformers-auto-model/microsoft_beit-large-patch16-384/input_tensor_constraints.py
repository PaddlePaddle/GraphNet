from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 384}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, S0, S0], "L_pixel_values_"),
    (
        [1024],
        "L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_",
    ),
    (
        [1024, 3, 16, 16],
        "L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_",
    ),
    ([1, 1, 1024], "L_self_modules_embeddings_parameters_cls_token_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_0_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_0_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_10_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_10_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_11_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_11_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_12_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_12_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_13_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_13_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_14_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_14_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_15_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_15_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_16_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_16_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_17_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_17_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_18_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_18_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_19_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_19_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_1_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_1_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_20_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_20_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_21_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_21_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_22_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_22_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_23_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_23_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_2_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_2_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_3_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_3_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_4_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_4_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_5_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_5_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_6_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_6_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_7_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_7_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_8_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_8_parameters_lambda_2_"),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [2212, 16],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_relative_position_bias_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024], "L_self_modules_encoder_modules_layer_modules_9_parameters_lambda_1_"),
    ([1024], "L_self_modules_encoder_modules_layer_modules_9_parameters_lambda_2_"),
    ([1024], "L_self_modules_pooler_modules_layernorm_parameters_bias_"),
    ([1024], "L_self_modules_pooler_modules_layernorm_parameters_weight_"),
]
