dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 20], "L_kwargs_attention_mask_"),
    ([1, 20], "L_kwargs_input_ids_"),
    ([], "L_self_modules_model_modules_embed_tokens_buffers_embed_scale_"),
    ([262144, 1152], "L_self_modules_model_modules_embed_tokens_parameters_weight_"),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_0_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_0_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_0_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_0_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_10_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_10_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_10_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_10_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_11_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_11_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_11_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_11_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_12_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_12_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_12_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_12_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_13_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_13_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_13_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_13_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_14_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_14_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_14_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_14_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_15_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_15_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_15_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_15_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_16_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_16_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_16_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_16_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_16_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_17_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_17_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_17_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_17_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_17_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_18_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_18_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_18_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_18_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_18_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_19_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_19_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_19_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_19_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_19_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_1_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_1_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_1_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_1_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_20_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_20_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_20_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_20_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_20_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_21_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_21_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_21_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_21_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_21_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_22_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_22_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_22_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_22_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_22_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_23_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_23_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_23_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_23_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_23_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_24_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_24_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_24_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_24_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_24_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_24_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_25_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_25_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_25_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_25_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_25_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_25_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_2_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_2_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_2_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_2_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_3_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_3_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_3_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_3_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_4_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_4_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_4_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_4_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_5_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_5_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_5_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_5_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_6_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_6_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_6_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_6_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_7_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_7_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_7_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_7_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_8_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_8_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_8_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_8_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_9_modules_input_layernorm_parameters_weight_",
    ),
    (
        [1152, 6912],
        "L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_down_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_gate_proj_parameters_weight_",
    ),
    (
        [6912, 1152],
        "L_self_modules_model_modules_layers_modules_9_modules_mlp_modules_up_proj_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_9_modules_post_feedforward_layernorm_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_model_modules_layers_modules_9_modules_pre_feedforward_layernorm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_norm_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1152, 1024],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_norm_parameters_weight_",
    ),
    (
        [1024, 1152],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256, 1152],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([1152], "L_self_modules_model_modules_norm_parameters_weight_"),
    ([128], "L_self_modules_model_modules_rotary_emb_buffers_inv_freq_"),
    ([128], "L_self_modules_model_modules_rotary_emb_local_buffers_inv_freq_"),
]
