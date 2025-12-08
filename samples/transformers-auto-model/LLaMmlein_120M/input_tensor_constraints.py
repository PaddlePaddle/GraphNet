dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 32], "L_kwargs_attention_mask_"),
    ([1, 32], "L_kwargs_input_ids_"),
    ([32064, 768], "L_self_modules_lm_head_parameters_weight_"),
    ([32064, 768], "L_self_modules_model_modules_embed_tokens_parameters_weight_"),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_",
    ),
    (
        [768, 2048],
        "L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_model_modules_norm_parameters_weight_"),
    ([32], "L_self_modules_model_modules_rotary_emb_buffers_inv_freq_"),
]
