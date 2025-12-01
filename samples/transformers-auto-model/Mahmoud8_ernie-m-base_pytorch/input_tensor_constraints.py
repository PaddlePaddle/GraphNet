dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 15], "L_input_ids_"),
    ([768], "L_self_modules_embeddings_modules_layer_norm_parameters_bias_"),
    ([768], "L_self_modules_embeddings_modules_layer_norm_parameters_weight_"),
    (
        [514, 768],
        "L_self_modules_embeddings_modules_position_embeddings_parameters_weight_",
    ),
    (
        [250002, 768],
        "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_0_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_10_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_10_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_11_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_11_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_1_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_2_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_3_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_4_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_5_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_5_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_6_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_6_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_7_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_7_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_8_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_8_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_9_modules_linear1_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_linear1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_linear2_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_9_modules_linear2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_norm1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_norm1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_norm2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_norm2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([768], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([768, 768], "L_self_modules_pooler_modules_dense_parameters_weight_"),
]
