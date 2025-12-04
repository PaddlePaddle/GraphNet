dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 2], "L_attention_mask_"),
    ([1, 2, 2560], "L_inputs_embeds_"),
    ([2560], "L_self_modules_final_layernorm_parameters_bias_"),
    ([2560], "L_self_modules_final_layernorm_parameters_weight_"),
    (
        [2560],
        "L_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_16_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_16_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_17_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_17_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_18_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_18_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_19_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_19_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_20_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_20_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_21_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_21_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_22_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_22_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_23_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_23_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_24_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_24_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_24_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_24_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_24_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_24_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_24_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_24_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_25_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_25_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_25_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_25_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_25_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_25_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_25_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_25_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_26_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_26_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_26_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_26_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_26_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_26_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_26_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_26_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_26_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_26_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_26_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_27_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_27_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_27_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_27_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_27_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_27_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_27_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_27_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_27_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_27_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_27_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_28_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_28_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_28_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_28_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_28_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_28_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_28_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_28_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_28_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_28_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_28_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_29_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_29_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_29_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_29_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_29_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_29_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_29_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_29_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_29_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_29_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_29_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_30_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_30_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_30_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_30_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_30_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_30_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_30_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_30_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_30_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_30_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_30_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_31_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_31_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_31_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_31_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_31_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_31_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_31_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_31_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_31_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_31_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_31_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_",
    ),
    (
        [10240],
        "L_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [10240, 2560],
        "L_self_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [2560, 10240],
        "L_self_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_dense_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [2560, 2560],
        "L_self_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([16], "L_self_modules_rotary_emb_buffers_inv_freq_"),
]
