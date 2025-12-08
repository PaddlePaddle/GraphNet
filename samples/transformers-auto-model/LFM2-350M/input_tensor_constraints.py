dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 20], "L_kwargs_attention_mask_"),
    ([1, 20], "L_kwargs_input_ids_"),
    ([65536, 1024], "L_self_modules_model_modules_embed_tokens_parameters_weight_"),
    ([1024], "L_self_modules_model_modules_embedding_norm_parameters_weight_"),
    (
        [1024, 1, 3],
        "L_self_modules_model_modules_layers_modules_0_modules_conv_modules_conv_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_model_modules_layers_modules_0_modules_conv_modules_in_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_0_modules_conv_modules_out_proj_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_0_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_0_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_0_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_0_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_0_modules_operator_norm_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_10_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_10_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_10_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_10_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_10_modules_operator_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_layernorm_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_layernorm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 1, 3],
        "L_self_modules_model_modules_layers_modules_11_modules_conv_modules_conv_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_model_modules_layers_modules_11_modules_conv_modules_in_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_11_modules_conv_modules_out_proj_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_11_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_11_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_11_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_11_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_11_modules_operator_norm_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_12_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_12_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_12_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_12_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_12_modules_operator_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_layernorm_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_layernorm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 1, 3],
        "L_self_modules_model_modules_layers_modules_13_modules_conv_modules_conv_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_model_modules_layers_modules_13_modules_conv_modules_in_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_13_modules_conv_modules_out_proj_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_13_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_13_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_13_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_13_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_13_modules_operator_norm_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_14_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_14_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_14_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_14_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_14_modules_operator_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_layernorm_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_layernorm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 1, 3],
        "L_self_modules_model_modules_layers_modules_15_modules_conv_modules_conv_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_model_modules_layers_modules_15_modules_conv_modules_in_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_15_modules_conv_modules_out_proj_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_15_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_15_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_15_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_15_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_15_modules_operator_norm_parameters_weight_",
    ),
    (
        [1024, 1, 3],
        "L_self_modules_model_modules_layers_modules_1_modules_conv_modules_conv_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_model_modules_layers_modules_1_modules_conv_modules_in_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_1_modules_conv_modules_out_proj_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_1_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_1_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_1_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_1_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_1_modules_operator_norm_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_2_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_2_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_2_modules_operator_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_layernorm_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_layernorm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 1, 3],
        "L_self_modules_model_modules_layers_modules_3_modules_conv_modules_conv_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_model_modules_layers_modules_3_modules_conv_modules_in_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_3_modules_conv_modules_out_proj_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_3_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_3_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_3_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_3_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_3_modules_operator_norm_parameters_weight_",
    ),
    (
        [1024, 1, 3],
        "L_self_modules_model_modules_layers_modules_4_modules_conv_modules_conv_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_model_modules_layers_modules_4_modules_conv_modules_in_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_4_modules_conv_modules_out_proj_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_4_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_4_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_4_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_4_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_4_modules_operator_norm_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_5_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_5_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_5_modules_operator_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_layernorm_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_layernorm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 1, 3],
        "L_self_modules_model_modules_layers_modules_6_modules_conv_modules_conv_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_model_modules_layers_modules_6_modules_conv_modules_in_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_6_modules_conv_modules_out_proj_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_6_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_6_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_6_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_6_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_6_modules_operator_norm_parameters_weight_",
    ),
    (
        [1024, 1, 3],
        "L_self_modules_model_modules_layers_modules_7_modules_conv_modules_conv_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_model_modules_layers_modules_7_modules_conv_modules_in_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_7_modules_conv_modules_out_proj_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_7_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_7_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_7_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_7_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_7_modules_operator_norm_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_8_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_8_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_8_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_8_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_8_modules_operator_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_layernorm_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_layernorm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [512, 1024],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 1, 3],
        "L_self_modules_model_modules_layers_modules_9_modules_conv_modules_conv_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_model_modules_layers_modules_9_modules_conv_modules_in_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_9_modules_conv_modules_out_proj_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_9_modules_feed_forward_modules_w1_parameters_weight_",
    ),
    (
        [1024, 4608],
        "L_self_modules_model_modules_layers_modules_9_modules_feed_forward_modules_w2_parameters_weight_",
    ),
    (
        [4608, 1024],
        "L_self_modules_model_modules_layers_modules_9_modules_feed_forward_modules_w3_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_9_modules_ffn_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_9_modules_operator_norm_parameters_weight_",
    ),
    ([32], "L_self_modules_model_modules_pos_emb_buffers_inv_freq_"),
]
