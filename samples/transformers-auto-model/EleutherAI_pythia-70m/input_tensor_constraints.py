dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 19], "L_kwargs_attention_mask_"),
    ([1, 19], "L_kwargs_input_ids_"),
    ([50304, 512], "L_self_modules_embed_out_parameters_weight_"),
    ([50304, 512], "L_self_modules_gpt_neox_modules_embed_in_parameters_weight_"),
    ([512], "L_self_modules_gpt_neox_modules_final_layer_norm_parameters_bias_"),
    ([512], "L_self_modules_gpt_neox_modules_final_layer_norm_parameters_weight_"),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_input_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_input_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_input_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_input_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_input_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [1536, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_input_layernorm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_gpt_neox_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_",
    ),
    ([8], "L_self_modules_gpt_neox_modules_rotary_emb_buffers_inv_freq_"),
]
