from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 336}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([2, 7], "L_kwargs_attention_mask_"),
    ([2, 7], "L_kwargs_input_ids_"),
    ([1, 3, S0, S0], "L_kwargs_pixel_values_"),
    ([1, 77], "L_self_modules_text_model_modules_embeddings_buffers_position_ids_"),
    (
        [77, 768],
        "L_self_modules_text_model_modules_embeddings_modules_position_embedding_parameters_weight_",
    ),
    (
        [49408, 768],
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
    ([768, 768], "L_self_modules_text_projection_parameters_weight_"),
    ([1, 577], "L_self_modules_vision_model_modules_embeddings_buffers_position_ids_"),
    (
        [1024, 3, 14, 14],
        "L_self_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_",
    ),
    (
        [577, 1024],
        "L_self_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_embeddings_parameters_class_embedding_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([1024], "L_self_modules_vision_model_modules_post_layernorm_parameters_bias_"),
    ([1024], "L_self_modules_vision_model_modules_post_layernorm_parameters_weight_"),
    ([1024], "L_self_modules_vision_model_modules_pre_layrnorm_parameters_bias_"),
    ([1024], "L_self_modules_vision_model_modules_pre_layrnorm_parameters_weight_"),
    ([768, 1024], "L_self_modules_visual_projection_parameters_weight_"),
    ([], "L_self_parameters_logit_scale_"),
]
