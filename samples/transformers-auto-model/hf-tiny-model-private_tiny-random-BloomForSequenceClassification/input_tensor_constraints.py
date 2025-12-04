dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 19], "L_attention_mask_"),
    ([1, 19, 32], "L_inputs_embeds_"),
    ([32], "L_self_modules_h_modules_0_modules_input_layernorm_parameters_bias_"),
    ([32], "L_self_modules_h_modules_0_modules_input_layernorm_parameters_weight_"),
    (
        [32],
        "L_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [32, 128],
        "L_self_modules_h_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [128, 32],
        "L_self_modules_h_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_h_modules_0_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_0_modules_self_attention_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_h_modules_0_modules_self_attention_modules_query_key_value_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_1_modules_input_layernorm_parameters_bias_"),
    ([32], "L_self_modules_h_modules_1_modules_input_layernorm_parameters_weight_"),
    (
        [32],
        "L_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [32, 128],
        "L_self_modules_h_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [128, 32],
        "L_self_modules_h_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_h_modules_1_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_1_modules_self_attention_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_h_modules_1_modules_self_attention_modules_query_key_value_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_2_modules_input_layernorm_parameters_bias_"),
    ([32], "L_self_modules_h_modules_2_modules_input_layernorm_parameters_weight_"),
    (
        [32],
        "L_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [32, 128],
        "L_self_modules_h_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [128, 32],
        "L_self_modules_h_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_h_modules_2_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_2_modules_self_attention_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_h_modules_2_modules_self_attention_modules_query_key_value_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_3_modules_input_layernorm_parameters_bias_"),
    ([32], "L_self_modules_h_modules_3_modules_input_layernorm_parameters_weight_"),
    (
        [32],
        "L_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [32, 128],
        "L_self_modules_h_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [128, 32],
        "L_self_modules_h_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_h_modules_3_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_3_modules_self_attention_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_h_modules_3_modules_self_attention_modules_query_key_value_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_4_modules_input_layernorm_parameters_bias_"),
    ([32], "L_self_modules_h_modules_4_modules_input_layernorm_parameters_weight_"),
    (
        [32],
        "L_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [32, 128],
        "L_self_modules_h_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [128, 32],
        "L_self_modules_h_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_h_modules_4_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_4_modules_self_attention_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_h_modules_4_modules_self_attention_modules_query_key_value_parameters_weight_",
    ),
    ([32], "L_self_modules_ln_f_parameters_bias_"),
    ([32], "L_self_modules_ln_f_parameters_weight_"),
    ([32], "L_self_modules_word_embeddings_layernorm_parameters_bias_"),
    ([32], "L_self_modules_word_embeddings_layernorm_parameters_weight_"),
]
