from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([2, 7], "L_kwargs_input_ids_"),
    ([1, 3, S0, S0], "L_kwargs_pixel_values_"),
    ([1, 64], "L_self_modules_text_model_modules_embeddings_buffers_position_ids_"),
    (
        [64, 768],
        "L_self_modules_text_model_modules_embeddings_modules_position_embedding_parameters_weight_",
    ),
    (
        [256000, 768],
        "L_self_modules_text_model_modules_embeddings_modules_token_embedding_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_text_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_text_model_modules_final_layer_norm_parameters_bias_"),
    ([768], "L_self_modules_text_model_modules_final_layer_norm_parameters_weight_"),
    ([768], "L_self_modules_text_model_modules_head_parameters_bias_"),
    ([768, 768], "L_self_modules_text_model_modules_head_parameters_weight_"),
    ([1, 196], "L_self_modules_vision_model_modules_embeddings_buffers_position_ids_"),
    (
        [768],
        "L_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_bias_",
    ),
    (
        [768, 3, 16, 16],
        "L_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_",
    ),
    (
        [196, 768],
        "L_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_head_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_vision_model_modules_head_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_vision_model_modules_head_modules_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_vision_model_modules_head_modules_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_head_modules_layernorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_head_modules_layernorm_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_vision_model_modules_head_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_vision_model_modules_head_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_vision_model_modules_head_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_vision_model_modules_head_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([1, 1, 768], "L_self_modules_vision_model_modules_head_parameters_probe_"),
    ([768], "L_self_modules_vision_model_modules_post_layernorm_parameters_bias_"),
    ([768], "L_self_modules_vision_model_modules_post_layernorm_parameters_weight_"),
    ([1], "L_self_parameters_logit_bias_"),
    ([1], "L_self_parameters_logit_scale_"),
]
