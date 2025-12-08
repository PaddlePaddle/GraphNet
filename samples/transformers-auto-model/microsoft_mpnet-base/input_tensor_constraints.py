from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 11], "L_attention_mask_"),
    ([S0, 11], "L_input_ids_"),
    ([768], "L_self_modules_embeddings_modules_LayerNorm_parameters_bias_"),
    ([768], "L_self_modules_embeddings_modules_LayerNorm_parameters_weight_"),
    (
        [514, 768],
        "L_self_modules_embeddings_modules_position_embeddings_parameters_weight_",
    ),
    (
        [30527, 768],
        "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32, 12],
        "L_self_modules_encoder_modules_relative_attention_bias_parameters_weight_",
    ),
    ([768], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([768, 768], "L_self_modules_pooler_modules_dense_parameters_weight_"),
]
