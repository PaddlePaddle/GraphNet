dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 36], "L_attention_mask_"),
    ([1, 36], "L_input_ids_"),
    ([32768], "L_self_modules_lm_head_parameters_bias_"),
    ([32768, 2048], "L_self_modules_lm_head_parameters_weight_"),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_0_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_0_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_10_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_10_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_11_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_11_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_1_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_1_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_2_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_2_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_3_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_3_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_4_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_4_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_5_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_5_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_6_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_6_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_7_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_7_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_8_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_8_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    (
        [2048, 64],
        "L_self_modules_transformer_modules_h_modules_9_modules_attn_embed_positions",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_transformer_modules_h_modules_9_modules_attn_scale_attn"),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    ([2048], "L_self_modules_transformer_modules_ln_f_parameters_bias_"),
    ([2048], "L_self_modules_transformer_modules_ln_f_parameters_weight_"),
    ([32768, 2048], "L_self_modules_transformer_modules_wte_parameters_weight_"),
]
