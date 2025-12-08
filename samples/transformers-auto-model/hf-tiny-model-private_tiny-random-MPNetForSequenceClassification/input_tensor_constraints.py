from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 45], "L_attention_mask_"),
    ([S0, 45], "L_input_ids_"),
    ([64], "L_self_modules_embeddings_modules_LayerNorm_parameters_bias_"),
    ([64], "L_self_modules_embeddings_modules_LayerNorm_parameters_weight_"),
    (
        [512, 64],
        "L_self_modules_embeddings_modules_position_embeddings_parameters_weight_",
    ),
    (
        [1125, 64],
        "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_LayerNorm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_o_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [64, 64],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32, 4],
        "L_self_modules_encoder_modules_relative_attention_bias_parameters_weight_",
    ),
    ([64], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([64, 64], "L_self_modules_pooler_modules_dense_parameters_weight_"),
]
