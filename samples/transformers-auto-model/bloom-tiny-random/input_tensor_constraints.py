dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 18], "L_attention_mask_"),
    ([1, 18], "L_input_ids_"),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_0_modules_input_layernorm_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [8, 32],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [32, 8],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_",
    ),
    (
        [8, 8],
        "L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [24, 8],
        "L_self_modules_transformer_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_1_modules_input_layernorm_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [8, 32],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [32, 8],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_",
    ),
    (
        [8, 8],
        "L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [24, 8],
        "L_self_modules_transformer_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_",
    ),
    ([8], "L_self_modules_transformer_modules_ln_f_parameters_bias_"),
    ([8], "L_self_modules_transformer_modules_ln_f_parameters_weight_"),
    (
        [8],
        "L_self_modules_transformer_modules_word_embeddings_layernorm_parameters_bias_",
    ),
    (
        [8],
        "L_self_modules_transformer_modules_word_embeddings_layernorm_parameters_weight_",
    ),
    (
        [250880, 8],
        "L_self_modules_transformer_modules_word_embeddings_parameters_weight_",
    ),
]
