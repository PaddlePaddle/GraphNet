dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 13], "L_attention_mask_"),
    ([1, 13], "L_input_ids_"),
    (
        [3072],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_0_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_v_"),
    (
        [3072],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_10_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_v_"),
    (
        [3072],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_11_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_v_"),
    (
        [3072],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_1_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_v_"),
    (
        [3072],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_2_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_v_"),
    (
        [3072],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_3_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_v_"),
    (
        [3072],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_4_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_v_"),
    (
        [3072],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_5_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_v_"),
    (
        [3072],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_6_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_v_"),
    (
        [3072],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_7_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_v_"),
    (
        [3072],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_8_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_v_"),
    (
        [3072],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_k_"),
    ([768, 12, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_o_"),
    ([768, 12, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_q_"),
    ([768, 12, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_"),
    ([12, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_r_bias_"),
    ([12, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_s_bias_"),
    ([12, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 12, 64],
        "L_self_modules_layer_modules_9_modules_rel_attn_parameters_seg_embed_",
    ),
    ([768, 12, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_v_"),
    ([32000, 768], "L_self_modules_word_embedding_parameters_weight_"),
    ([1, 13], "L_token_type_ids_"),
]
