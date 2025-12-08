from sympy import Symbol, Expr, Rel, Eq


dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 13], "L_attention_mask_"),
    ([1, 13, 768], "L_inputs_embeds_"),
    ([2050, 768], "L_self_modules_embed_positions_parameters_weight_"),
    ([768], "L_self_modules_final_layer_norm_parameters_bias_"),
    ([768], "L_self_modules_final_layer_norm_parameters_weight_"),
    ([3072], "L_self_modules_layers_modules_0_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_0_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_0_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_0_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([3072], "L_self_modules_layers_modules_10_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_10_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_10_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_10_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([3072], "L_self_modules_layers_modules_11_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_11_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_11_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_11_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([3072], "L_self_modules_layers_modules_1_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_1_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_1_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_1_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([3072], "L_self_modules_layers_modules_2_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_2_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_2_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_2_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([3072], "L_self_modules_layers_modules_3_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_3_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_3_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_3_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([3072], "L_self_modules_layers_modules_4_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_4_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_4_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_4_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([3072], "L_self_modules_layers_modules_5_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_5_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_5_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_5_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([3072], "L_self_modules_layers_modules_6_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_6_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_6_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_6_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([3072], "L_self_modules_layers_modules_7_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_7_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_7_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_7_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([3072], "L_self_modules_layers_modules_8_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_8_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_8_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_8_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([3072], "L_self_modules_layers_modules_9_modules_fc1_parameters_bias_"),
    ([3072, 768], "L_self_modules_layers_modules_9_modules_fc1_parameters_weight_"),
    ([768], "L_self_modules_layers_modules_9_modules_fc2_parameters_bias_"),
    ([768, 3072], "L_self_modules_layers_modules_9_modules_fc2_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
]
