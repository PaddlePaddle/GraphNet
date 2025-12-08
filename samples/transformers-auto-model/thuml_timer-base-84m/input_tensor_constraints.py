from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 2880}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, S0], "L_input_ids_"),
    ([96, 1024], "L_self_modules_lm_heads_modules_0_parameters_weight_"),
    (
        [1024, 96],
        "L_self_modules_model_modules_embed_layer_modules_emb_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 2048],
        "L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [2048, 1024],
        "L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 128],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([1024], "L_self_modules_model_modules_norm_parameters_bias_"),
    ([1024], "L_self_modules_model_modules_norm_parameters_weight_"),
]
