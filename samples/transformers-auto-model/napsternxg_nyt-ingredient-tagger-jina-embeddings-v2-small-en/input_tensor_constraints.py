dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 11], "L_attention_mask_"),
    ([1, 11], "L_input_ids_"),
    ([512], "L_self_modules_embeddings_modules_LayerNorm_parameters_bias_"),
    ([512], "L_self_modules_embeddings_modules_LayerNorm_parameters_weight_"),
    (
        [2, 512],
        "L_self_modules_embeddings_modules_token_type_embeddings_parameters_weight_",
    ),
    (
        [30528, 512],
        "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    ([1, 8, 8192, 8192], "L_self_modules_encoder_buffers_alibi_"),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096, 512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_gated_layers_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layer_modules_0_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096, 512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_gated_layers_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layer_modules_1_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096, 512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_gated_layers_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layer_modules_2_modules_mlp_modules_wo_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [4096, 512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_gated_layers_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_encoder_modules_layer_modules_3_modules_mlp_modules_wo_parameters_weight_",
    ),
    ([512], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([512, 512], "L_self_modules_pooler_modules_dense_parameters_weight_"),
    ([1, 11], "L_token_type_ids_"),
]
