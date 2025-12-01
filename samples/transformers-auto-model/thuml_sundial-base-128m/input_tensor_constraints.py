from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 2880}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, S0], "L_input_ids_"),
    ([768], "L_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_bias_"),
    (
        [768, 768],
        "L_self_modules_flow_loss_modules_net_modules_cond_embed_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_flow_loss_modules_net_modules_final_layer_modules_adaLN_modulation_modules_1_parameters_bias_",
    ),
    (
        [1536, 768],
        "L_self_modules_flow_loss_modules_net_modules_final_layer_modules_adaLN_modulation_modules_1_parameters_weight_",
    ),
    (
        [720],
        "L_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_bias_",
    ),
    (
        [720, 768],
        "L_self_modules_flow_loss_modules_net_modules_final_layer_modules_linear_parameters_weight_",
    ),
    ([768], "L_self_modules_flow_loss_modules_net_modules_input_proj_parameters_bias_"),
    (
        [768, 720],
        "L_self_modules_flow_loss_modules_net_modules_input_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_adaLN_modulation_modules_1_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_adaLN_modulation_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_in_ln_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_0_modules_mlp_modules_2_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_adaLN_modulation_modules_1_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_adaLN_modulation_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_in_ln_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_1_modules_mlp_modules_2_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_adaLN_modulation_modules_1_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_adaLN_modulation_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_in_ln_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_flow_loss_modules_net_modules_res_blocks_modules_2_modules_mlp_modules_2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_flow_loss_modules_net_modules_time_embed_modules_mlp_modules_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_bias_",
    ),
    (
        [3072, 32],
        "L_self_modules_model_modules_embed_layer_modules_hidden_layer_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_embed_layer_modules_output_layer_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_embed_layer_modules_output_layer_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_bias_",
    ),
    (
        [768, 32],
        "L_self_modules_model_modules_embed_layer_modules_residual_layer_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_10_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_10_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_11_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_11_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_6_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_6_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_7_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_7_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_8_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_8_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_down_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_gate_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_ffn_layer_modules_up_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_9_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_9_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_o_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_cos_cached_",
    ),
    (
        [10000, 64],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_rotary_emb_buffers_sin_cached_",
    ),
    (
        [768],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_model_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_model_modules_norm_parameters_bias_"),
    ([768], "L_self_modules_model_modules_norm_parameters_weight_"),
]
