dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_layernorm_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_decoder_modules_decoder_modules_decoding_cross_attention_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [3],
        "L_self_modules_decoder_modules_decoder_modules_final_layer_parameters_bias_",
    ),
    (
        [3, 20],
        "L_self_modules_decoder_modules_decoder_modules_final_layer_parameters_weight_",
    ),
    (
        [1, 20],
        "L_self_modules_decoder_modules_decoder_modules_output_position_encodings_parameters_position_embeddings_",
    ),
    ([10, 20], "L_self_modules_embeddings_parameters_latents_"),
    (
        [20],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [20, 261],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm1_parameters_weight_",
    ),
    (
        [261],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_bias_",
    ),
    (
        [261],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_layernorm2_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [20, 261],
        "L_self_modules_encoder_modules_cross_attention_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_cross_attention_modules_layernorm_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [80, 20],
        "L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [20, 80],
        "L_self_modules_encoder_modules_cross_attention_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_layernorm1_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_layernorm_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [80, 20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [20, 80],
        "L_self_modules_encoder_modules_self_attends_modules_0_modules_mlp_modules_dense2_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_layernorm1_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [20, 20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_bias_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_layernorm_parameters_weight_",
    ),
    (
        [80],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_bias_",
    ),
    (
        [80, 20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense1_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_bias_",
    ),
    (
        [20, 80],
        "L_self_modules_encoder_modules_self_attends_modules_1_modules_mlp_modules_dense2_parameters_weight_",
    ),
    ([1, 1024, 261], "L_stack0_0_"),
]
