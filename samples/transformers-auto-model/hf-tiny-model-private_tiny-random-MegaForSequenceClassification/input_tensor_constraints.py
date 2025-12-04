dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 22], "L_attention_mask_"),
    ([1, 22], "L_input_ids_"),
    ([1, 1024], "L_self_modules_embedding_layer_buffers_token_type_ids_"),
    (
        [2, 32],
        "L_self_modules_embedding_layer_modules_token_type_embeddings_parameters_weight_",
    ),
    (
        [1024, 32],
        "L_self_modules_embedding_layer_modules_word_embeddings_parameters_weight_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_damping_factor_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_decay_factor_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_",
    ),
    (
        [32, 16],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_",
    ),
    (
        [32],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_ema_gate_parameters_residual_weight_",
    ),
    (
        [32],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_h_proj_parameters_weight_",
    ),
    (
        [165],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_bias_",
    ),
    (
        [165, 32],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_mx_proj_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_",
    ),
    (
        [1024, 32],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_cosine",
    ),
    (
        [1, 64],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_",
    ),
    (
        [1, 64],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_",
    ),
    (
        [1024, 32],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_rel_pos_bias_sine",
    ),
    (
        [37],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_layers_modules_0_modules_mega_layer_modules_v_proj_parameters_weight_",
    ),
    ([2, 64], "L_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_bias_"),
    (
        [2, 64],
        "L_self_modules_layers_modules_0_modules_mega_layer_parameters_qk_weight_",
    ),
    ([24], "L_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_bias_"),
    (
        [24, 32],
        "L_self_modules_layers_modules_0_modules_nffn_modules_fc1_parameters_weight_",
    ),
    ([32], "L_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_bias_"),
    (
        [32, 24],
        "L_self_modules_layers_modules_0_modules_nffn_modules_fc2_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_0_modules_nffn_modules_norm_modules_norm_parameters_scalar_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_damping_factor_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_decay_factor_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_",
    ),
    (
        [32, 16],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_",
    ),
    (
        [32],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_ema_gate_parameters_residual_weight_",
    ),
    (
        [32],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_h_proj_parameters_weight_",
    ),
    (
        [165],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_bias_",
    ),
    (
        [165, 32],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_mx_proj_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_",
    ),
    (
        [1024, 32],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_cosine",
    ),
    (
        [1, 64],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_",
    ),
    (
        [1, 64],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_",
    ),
    (
        [1024, 32],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_rel_pos_bias_sine",
    ),
    (
        [37],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_layers_modules_1_modules_mega_layer_modules_v_proj_parameters_weight_",
    ),
    ([2, 64], "L_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_bias_"),
    (
        [2, 64],
        "L_self_modules_layers_modules_1_modules_mega_layer_parameters_qk_weight_",
    ),
    ([24], "L_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_bias_"),
    (
        [24, 32],
        "L_self_modules_layers_modules_1_modules_nffn_modules_fc1_parameters_weight_",
    ),
    ([32], "L_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_bias_"),
    (
        [32, 24],
        "L_self_modules_layers_modules_1_modules_nffn_modules_fc2_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_1_modules_nffn_modules_norm_modules_norm_parameters_scalar_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_damping_factor_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_decay_factor_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_",
    ),
    (
        [32, 16],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_",
    ),
    (
        [32],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_ema_gate_parameters_residual_weight_",
    ),
    (
        [32],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_h_proj_parameters_weight_",
    ),
    (
        [165],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_bias_",
    ),
    (
        [165, 32],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_mx_proj_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_",
    ),
    (
        [1024, 32],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_cosine",
    ),
    (
        [1, 64],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_",
    ),
    (
        [1, 64],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_",
    ),
    (
        [1024, 32],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_rel_pos_bias_sine",
    ),
    (
        [37],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_layers_modules_2_modules_mega_layer_modules_v_proj_parameters_weight_",
    ),
    ([2, 64], "L_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_bias_"),
    (
        [2, 64],
        "L_self_modules_layers_modules_2_modules_mega_layer_parameters_qk_weight_",
    ),
    ([24], "L_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_bias_"),
    (
        [24, 32],
        "L_self_modules_layers_modules_2_modules_nffn_modules_fc1_parameters_weight_",
    ),
    ([32], "L_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_bias_"),
    (
        [32, 24],
        "L_self_modules_layers_modules_2_modules_nffn_modules_fc2_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_2_modules_nffn_modules_norm_modules_norm_parameters_scalar_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_damping_factor_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_decay_factor_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_",
    ),
    (
        [32, 16],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_",
    ),
    (
        [32],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_ema_gate_parameters_residual_weight_",
    ),
    (
        [32],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_h_proj_parameters_weight_",
    ),
    (
        [165],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_bias_",
    ),
    (
        [165, 32],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_mx_proj_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_",
    ),
    (
        [1024, 32],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_cosine",
    ),
    (
        [1, 64],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_",
    ),
    (
        [1, 64],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_",
    ),
    (
        [1024, 32],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_rel_pos_bias_sine",
    ),
    (
        [37],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_layers_modules_3_modules_mega_layer_modules_v_proj_parameters_weight_",
    ),
    ([2, 64], "L_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_bias_"),
    (
        [2, 64],
        "L_self_modules_layers_modules_3_modules_mega_layer_parameters_qk_weight_",
    ),
    ([24], "L_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_bias_"),
    (
        [24, 32],
        "L_self_modules_layers_modules_3_modules_nffn_modules_fc1_parameters_weight_",
    ),
    ([32], "L_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_bias_"),
    (
        [32, 24],
        "L_self_modules_layers_modules_3_modules_nffn_modules_fc2_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_3_modules_nffn_modules_norm_modules_norm_parameters_scalar_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_damping_factor_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_decay_factor_",
    ),
    (
        [32, 16, 1],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_ema_expansion_matrix_",
    ),
    (
        [32, 16],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_kernel_projection_matrix_",
    ),
    (
        [32],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_ema_gate_parameters_residual_weight_",
    ),
    (
        [32],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_h_proj_parameters_weight_",
    ),
    (
        [165],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_bias_",
    ),
    (
        [165, 32],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_mx_proj_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_norm_modules_norm_parameters_scalar_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_buffers_float_tensor_",
    ),
    (
        [1024, 32],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_cosine",
    ),
    (
        [1, 64],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_alpha_",
    ),
    (
        [1, 64],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_parameters_b_param_",
    ),
    (
        [1024, 32],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_rel_pos_bias_sine",
    ),
    (
        [37],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_layers_modules_4_modules_mega_layer_modules_v_proj_parameters_weight_",
    ),
    ([2, 64], "L_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_bias_"),
    (
        [2, 64],
        "L_self_modules_layers_modules_4_modules_mega_layer_parameters_qk_weight_",
    ),
    ([24], "L_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_bias_"),
    (
        [24, 32],
        "L_self_modules_layers_modules_4_modules_nffn_modules_fc1_parameters_weight_",
    ),
    ([32], "L_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_bias_"),
    (
        [32, 24],
        "L_self_modules_layers_modules_4_modules_nffn_modules_fc2_parameters_weight_",
    ),
    (
        [1],
        "L_self_modules_layers_modules_4_modules_nffn_modules_norm_modules_norm_parameters_scalar_",
    ),
    ([32], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([32, 32], "L_self_modules_pooler_modules_dense_parameters_weight_"),
]
