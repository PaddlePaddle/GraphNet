from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 13], "L_attention_mask_"),
    ([S0, 13], "L_input_ids_"),
    (
        [4096],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_0_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_0_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_0_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_0_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_10_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_10_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_10_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_10_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_11_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_11_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_11_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_11_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_12_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_12_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_12_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_12_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_12_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_12_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_12_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_12_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_12_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_12_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_13_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_13_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_13_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_13_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_13_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_13_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_13_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_13_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_13_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_13_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_14_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_14_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_14_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_14_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_14_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_14_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_14_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_14_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_14_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_14_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_15_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_15_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_15_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_15_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_15_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_15_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_15_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_15_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_15_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_15_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_16_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_16_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_16_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_16_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_16_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_16_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_16_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_16_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_16_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_16_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_17_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_17_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_17_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_17_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_17_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_17_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_17_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_17_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_17_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_17_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_18_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_18_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_18_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_18_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_18_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_18_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_18_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_18_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_18_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_18_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_19_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_19_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_19_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_19_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_19_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_19_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_19_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_19_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_19_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_19_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_1_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_1_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_1_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_1_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_20_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_20_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_20_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_20_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_20_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_20_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_20_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_20_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_20_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_20_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_21_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_21_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_21_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_21_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_21_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_21_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_21_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_21_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_21_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_21_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_22_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_22_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_22_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_22_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_22_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_22_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_22_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_22_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_22_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_22_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_23_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_23_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_23_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_23_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_23_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_23_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_23_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_23_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_23_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_23_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_2_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_2_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_2_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_2_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_3_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_3_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_3_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_3_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_4_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_4_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_4_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_4_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_5_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_5_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_5_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_5_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_6_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_6_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_6_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_6_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_7_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_7_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_7_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_7_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_8_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_8_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_8_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_8_modules_rel_attn_parameters_v_"),
    (
        [4096],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_9_modules_ff_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer_modules_9_modules_rel_attn_modules_layer_norm_parameters_weight_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_k_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_o_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_q_"),
    ([1024, 16, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_"),
    ([16, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_r_bias_"),
    ([16, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_s_bias_"),
    ([16, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_r_w_bias_"),
    (
        [2, 16, 64],
        "L_self_modules_layer_modules_9_modules_rel_attn_parameters_seg_embed_",
    ),
    ([1024, 16, 64], "L_self_modules_layer_modules_9_modules_rel_attn_parameters_v_"),
    ([32000, 1024], "L_self_modules_word_embedding_parameters_weight_"),
    ([S0, 13], "L_token_type_ids_"),
]
