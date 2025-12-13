from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 16], "L_encoder_attention_mask_"),
    ([S0, 16, 1024], "L_encoder_hidden_states_"),
    ([S0, 1, 1024], "L_inputs_embeds_"),
    (
        [1024, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [32, 16],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_10_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_10_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_11_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_11_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_12_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_12_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_13_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_13_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_14_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_14_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_15_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_15_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_16_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_16_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_17_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_17_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_18_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_18_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_19_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_19_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_20_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_20_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_20_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_20_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_21_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_21_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_21_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_21_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_22_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_22_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_22_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_22_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_23_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_23_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_23_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_23_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_4_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_4_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_5_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_5_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_6_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_6_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_7_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_7_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_8_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_8_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [1024, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_2_modules_DenseReluDense_modules_wi_0_parameters_weight_",
    ),
    (
        [2816, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_2_modules_DenseReluDense_modules_wi_1_parameters_weight_",
    ),
    (
        [1024, 2816],
        "L_self_modules_block_modules_9_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_9_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    ([1024], "L_self_modules_final_layer_norm_parameters_weight_"),
]
