from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, S0, S0], "L_pixel_values_"),
    ([], "L_self_modules_embeddings_modules_dropout_p"),
    (
        [32],
        "L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_",
    ),
    (
        [32, 3, 16, 16],
        "L_self_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_",
    ),
    ([1, 1, 32], "L_self_modules_embeddings_parameters_cls_token_"),
    ([1, 197, 32], "L_self_modules_embeddings_parameters_position_embeddings_"),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    ([], "L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_eps"),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    ([], "L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_eps"),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 3072],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    ([], "L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_eps"),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    ([], "L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_eps"),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 3072],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p",
    ),
    ([], "L_self_modules_layernorm_eps"),
    ([32], "L_self_modules_layernorm_parameters_bias_"),
    ([32], "L_self_modules_layernorm_parameters_weight_"),
    ([32], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([32, 32], "L_self_modules_pooler_modules_dense_parameters_weight_"),
]
