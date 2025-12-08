from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 20}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_attention_mask_"),
    ([S0, S1], "L_input_ids_"),
    ([1, 512], "L_self_modules_embeddings_buffers_position_ids_"),
    ([128], "L_self_modules_embeddings_modules_LayerNorm_parameters_bias_"),
    ([128], "L_self_modules_embeddings_modules_LayerNorm_parameters_weight_"),
    (
        [512, 128],
        "L_self_modules_embeddings_modules_position_embeddings_parameters_weight_",
    ),
    (
        [2, 128],
        "L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_",
    ),
    (
        [30000, 128],
        "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_output_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_ffn_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_albert_layer_groups_modules_0_modules_albert_layers_modules_0_modules_full_layer_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_bias_",
    ),
    (
        [768, 128],
        "L_self_modules_encoder_modules_embedding_hidden_mapping_in_parameters_weight_",
    ),
    ([768], "L_self_modules_pooler_parameters_bias_"),
    ([768, 768], "L_self_modules_pooler_parameters_weight_"),
    ([S0, S1], "L_token_type_ids_"),
]
