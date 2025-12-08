from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 12], "L_attention_mask_"),
    ([S0, 12], "L_input_ids_"),
    (
        [192, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [32, 12],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 192],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [192, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    ([32128, 768], "L_self_modules_embed_tokens_parameters_weight_"),
    ([768], "L_self_modules_final_layer_norm_parameters_weight_"),
]
