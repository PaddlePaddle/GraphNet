from sympy import Symbol, Expr, Rel, Eq


dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 10], "L_attention_mask_"),
    ([1, 10, 1024], "L_inputs_embeds_"),
    ([1026, 1024], "L_self_modules_embed_positions_parameters_weight_"),
    ([1024], "L_self_modules_layer_norm_parameters_bias_"),
    ([1024], "L_self_modules_layer_norm_parameters_weight_"),
    ([4096], "L_self_modules_layers_modules_0_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_0_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_0_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_0_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_10_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_10_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_10_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_10_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_10_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_11_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_11_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_11_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_11_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_11_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_12_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_12_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_12_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_12_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_12_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_13_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_13_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_13_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_13_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_13_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_14_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_14_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_14_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_14_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_14_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_15_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_15_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_15_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_15_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_15_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_16_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_16_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_16_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_16_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_16_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_17_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_17_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_17_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_17_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_17_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_18_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_18_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_18_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_18_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_18_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_19_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_19_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_19_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_19_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_19_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_1_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_1_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_1_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_1_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_20_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_20_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_20_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_20_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_20_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_21_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_21_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_21_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_21_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_21_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_22_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_22_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_22_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_22_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_22_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_23_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_23_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_23_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_23_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_23_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_2_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_2_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_2_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_2_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_3_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_3_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_3_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_3_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_4_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_4_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_4_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_4_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_5_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_5_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_5_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_5_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_6_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_6_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_6_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_6_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_6_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_7_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_7_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_7_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_7_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_7_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_8_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_8_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_8_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_8_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_8_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4096], "L_self_modules_layers_modules_9_modules_fc1_parameters_bias_"),
    ([4096, 1024], "L_self_modules_layers_modules_9_modules_fc1_parameters_weight_"),
    ([1024], "L_self_modules_layers_modules_9_modules_fc2_parameters_bias_"),
    ([1024, 4096], "L_self_modules_layers_modules_9_modules_fc2_parameters_weight_"),
    (
        [1024],
        "L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_9_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
]
