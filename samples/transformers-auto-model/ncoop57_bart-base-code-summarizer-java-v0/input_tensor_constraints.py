from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 11], "L_encoder_attention_mask_"),
    ([S0, 11, 768], "L_encoder_hidden_states_"),
    ([S0, 11, 768], "L_inputs_embeds_"),
    ([1026, 768], "L_self_modules_embed_positions_parameters_weight_"),
    ([768], "L_self_modules_layernorm_embedding_parameters_bias_"),
    ([768], "L_self_modules_layernorm_embedding_parameters_weight_"),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
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
    (
        [768],
        "L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_",
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
    (
        [768],
        "L_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_",
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
    (
        [768],
        "L_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_",
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
    (
        [768],
        "L_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_",
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
    (
        [768],
        "L_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_",
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
]
