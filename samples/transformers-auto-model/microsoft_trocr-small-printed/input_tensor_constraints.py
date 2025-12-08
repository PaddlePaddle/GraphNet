from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 384}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 1], "L_decoder_input_ids_"),
    ([1, 3, S0, S0], "L_pixel_values_"),
    (
        [514, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_embed_positions_parameters_weight_",
    ),
    (
        [64044, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_embed_tokens_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layernorm_embedding_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layernorm_embedding_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 384],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_encoder_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_decoder_modules_model_modules_decoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [64044, 256],
        "L_self_modules_decoder_modules_output_projection_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_",
    ),
    (
        [384, 3, 16, 16],
        "L_self_modules_encoder_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_",
    ),
    ([1, 1, 384], "L_self_modules_encoder_modules_embeddings_parameters_cls_token_"),
    (
        [1, 1, 384],
        "L_self_modules_encoder_modules_embeddings_parameters_distillation_token_",
    ),
    (
        [1, 578, 384],
        "L_self_modules_encoder_modules_embeddings_parameters_position_embeddings_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [1536, 384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [384, 1536],
        "L_self_modules_encoder_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    ([384], "L_self_modules_encoder_modules_layernorm_parameters_bias_"),
    ([384], "L_self_modules_encoder_modules_layernorm_parameters_weight_"),
    ([384], "L_self_modules_encoder_modules_pooler_modules_dense_parameters_bias_"),
    (
        [384, 384],
        "L_self_modules_encoder_modules_pooler_modules_dense_parameters_weight_",
    ),
]
