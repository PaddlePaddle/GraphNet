from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 1], "L_decoder_input_ids_"),
    ([S0, 2048, 770], "L_flattened_patches_"),
    ([50244, 768], "L_self_modules_decoder_modules_embed_tokens_parameters_weight_"),
    ([768], "L_self_modules_decoder_modules_final_layer_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_0_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [32, 12],
        "L_self_modules_decoder_modules_layer_modules_0_modules_self_attention_modules_attention_modules_relative_attention_bias_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_0_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_10_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_10_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_11_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_11_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_1_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_1_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_2_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_2_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_3_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_3_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_4_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_4_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_5_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_5_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_6_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_6_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_7_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_7_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_8_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_8_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_encoder_decoder_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_encoder_decoder_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_encoder_decoder_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_encoder_decoder_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_encoder_decoder_attention_modules_layer_norm_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_mlp_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_mlp_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_decoder_modules_layer_modules_9_modules_mlp_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_mlp_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_self_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_self_attention_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_self_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_self_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decoder_modules_layer_modules_9_modules_self_attention_modules_layer_norm_parameters_weight_",
    ),
    ([50244, 768], "L_self_modules_decoder_modules_lm_head_parameters_weight_"),
    (
        [4096, 768],
        "L_self_modules_encoder_modules_embeddings_modules_column_embedder_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_embeddings_modules_patch_projection_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_embeddings_modules_patch_projection_parameters_weight_",
    ),
    (
        [4096, 768],
        "L_self_modules_encoder_modules_embeddings_modules_row_embedder_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_mlp_modules_wi_0_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_mlp_modules_wi_1_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_pre_attention_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_pre_mlp_layer_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_encoder_modules_layernorm_parameters_weight_"),
]
