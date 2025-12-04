dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 45], "L_attention_mask_"),
    ([1, 45], "L_input_ids_"),
    ([1, 512], "L_self_modules_embeddings_buffers_position_ids_"),
    ([32], "L_self_modules_embeddings_modules_LayerNorm_parameters_bias_"),
    ([32], "L_self_modules_embeddings_modules_LayerNorm_parameters_weight_"),
    (
        [512, 32],
        "L_self_modules_embeddings_modules_position_embeddings_parameters_weight_",
    ),
    (
        [1124, 32],
        "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_k_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_out_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_q_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_attention_modules_v_lin_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_transformer_modules_layer_modules_0_modules_ffn_modules_lin2_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_output_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_0_modules_sa_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_k_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_out_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_q_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_attention_modules_v_lin_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_transformer_modules_layer_modules_1_modules_ffn_modules_lin2_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_output_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_1_modules_sa_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_k_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_out_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_q_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_attention_modules_v_lin_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_transformer_modules_layer_modules_2_modules_ffn_modules_lin2_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_output_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_2_modules_sa_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_k_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_out_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_q_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_attention_modules_v_lin_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_transformer_modules_layer_modules_3_modules_ffn_modules_lin2_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_output_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_3_modules_sa_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_k_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_out_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_q_lin_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_attention_modules_v_lin_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_transformer_modules_layer_modules_4_modules_ffn_modules_lin2_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_output_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_transformer_modules_layer_modules_4_modules_sa_layer_norm_parameters_weight_",
    ),
]
