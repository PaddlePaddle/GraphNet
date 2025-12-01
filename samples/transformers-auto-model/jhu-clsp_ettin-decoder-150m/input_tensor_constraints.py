dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 10], "L_attention_mask_"),
    ([1, 10], "L_input_ids_"),
    ([768], "L_self_modules_embeddings_modules_norm_parameters_weight_"),
    (
        [50368, 768],
        "L_self_modules_embeddings_modules_tok_embeddings_parameters_weight_",
    ),
    ([768], "L_self_modules_final_norm_parameters_weight_"),
    ([32], "L_self_modules_global_rotary_emb_buffers_inv_freq_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2304, 768],
        "L_self_modules_layers_modules_0_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_0_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_0_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_10_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_10_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_10_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_10_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_10_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_10_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_10_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_10_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_11_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_11_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_11_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_11_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_11_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_11_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_11_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_11_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_12_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_12_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_12_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_12_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_12_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_12_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_12_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_12_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_13_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_13_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_13_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_13_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_13_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_13_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_13_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_13_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_14_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_14_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_14_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_14_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_14_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_14_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_14_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_14_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_15_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_15_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_15_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_15_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_15_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_15_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_15_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_15_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_16_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_16_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_16_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_16_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_16_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_16_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_16_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_16_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_17_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_17_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_17_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_17_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_17_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_17_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_17_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_17_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_18_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_18_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_18_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_18_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_18_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_18_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_18_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_18_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_19_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_19_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_19_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_19_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_19_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_19_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_19_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_19_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_1_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_1_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_1_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_1_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_20_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_20_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_20_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_20_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_20_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_20_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_20_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_20_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_21_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_21_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_21_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_21_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_21_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_21_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_21_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_21_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_2_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_2_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_2_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_2_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_3_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_3_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_3_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_3_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_4_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_4_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_4_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_4_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_5_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_5_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_5_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_5_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_6_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_6_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_6_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_6_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_6_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_6_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_6_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_6_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_7_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_7_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_7_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_7_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_7_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_7_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_7_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_7_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_8_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_8_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_8_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_8_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_8_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_8_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_8_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_8_modules_mlp_norm_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_layers_modules_9_modules_attn_modules_Wo_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_9_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_9_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_9_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_9_modules_attn_norm_parameters_weight_"),
    (
        [2304, 768],
        "L_self_modules_layers_modules_9_modules_mlp_modules_Wi_parameters_weight_",
    ),
    (
        [768, 1152],
        "L_self_modules_layers_modules_9_modules_mlp_modules_Wo_parameters_weight_",
    ),
    ([768], "L_self_modules_layers_modules_9_modules_mlp_norm_parameters_weight_"),
    ([32], "L_self_modules_local_rotary_emb_buffers_inv_freq_"),
]
