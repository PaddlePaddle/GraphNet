from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [768, 768],
        "L_self_modules_blocks_modules_0_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_0_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_0_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_blocks_modules_10_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_10_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_10_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_10_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_blocks_modules_11_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_11_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_11_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_11_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_blocks_modules_1_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_1_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_1_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_blocks_modules_2_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_2_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_2_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_blocks_modules_3_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_3_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_3_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_blocks_modules_4_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_4_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_4_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_blocks_modules_5_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_5_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_5_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_5_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_blocks_modules_6_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_6_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_6_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_6_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_blocks_modules_7_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_7_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_7_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_7_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_blocks_modules_8_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_8_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_8_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_8_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_"),
    (
        [768, 768],
        "L_self_modules_blocks_modules_9_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_9_modules_attn_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_9_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_blocks_modules_9_modules_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_g_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_bias_",
    ),
    (
        [2048, 768],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_x_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [768, 2048],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_"),
    ([768], "L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_"),
    ([512], "L_self_modules_head_parameters_bias_"),
    ([512, 768], "L_self_modules_head_parameters_weight_"),
    ([768], "L_self_modules_norm_parameters_bias_"),
    ([768], "L_self_modules_norm_parameters_weight_"),
    ([768], "L_self_modules_patch_embed_modules_proj_parameters_bias_"),
    ([768, 3, 16, 16], "L_self_modules_patch_embed_modules_proj_parameters_weight_"),
    ([196, 128], "L_self_modules_rope_buffers_pos_embed_"),
    ([1, 1, 768], "L_self_parameters_cls_token_"),
    ([1, 197, 768], "L_self_parameters_pos_embed_"),
    ([1, 3, S0, S0], "L_x_"),
]
