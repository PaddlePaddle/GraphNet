from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 42}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_attention_mask_"),
    ([S0, 1], "L_decoder_input_ids_"),
    ([S0, S1], "L_input_ids_"),
    ([1, 61871], "L_self_buffers_final_logits_bias_"),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_embed_positions_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_embed_positions_parameters_weight_",
    ),
    (
        [61871, 1024],
        "L_self_modules_model_modules_encoder_modules_embed_tokens_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
]
