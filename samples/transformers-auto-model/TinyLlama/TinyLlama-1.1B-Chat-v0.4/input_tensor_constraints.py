from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 3}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_kwargs_attention_mask_"),
    ([S0, S1], "L_kwargs_input_ids_"),
    ([32003, 2048], "L_self_modules_model_modules_embed_tokens_parameters_weight_"),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048, 5632],
        "L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [5632, 2048],
        "L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [2048, 2048],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 2048],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([2048], "L_self_modules_model_modules_norm_parameters_weight_"),
    ([32], "L_self_modules_model_modules_rotary_emb_buffers_inv_freq_"),
]
