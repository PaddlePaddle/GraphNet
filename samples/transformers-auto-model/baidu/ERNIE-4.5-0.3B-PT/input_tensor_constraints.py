from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 2], "L_attention_mask_"),
    ([S0, 2, 1024], "L_inputs_embeds_"),
    (
        [1024],
        "L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1024, 3072],
        "L_self_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1024],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([1024], "L_self_modules_norm_parameters_weight_"),
    ([64], "L_self_modules_rotary_emb_buffers_inv_freq_"),
]
