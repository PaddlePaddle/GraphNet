from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 2}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 6], "L_attention_mask_"),
    ([S0, 6], "L_input_ids_"),
    (
        [16384, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [32, 128],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_12_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_12_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_12_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_12_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_13_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_13_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_13_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_13_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_14_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_14_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_14_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_14_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_15_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_15_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_15_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_15_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_16_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_16_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_16_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_16_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_17_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_17_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_17_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_17_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_18_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_18_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_18_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_18_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_19_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_19_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_19_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_19_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_20_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_20_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_20_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_20_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_21_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_21_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_21_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_21_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_22_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_22_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_22_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_22_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_23_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_23_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_23_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_23_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [1024, 16384],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [16384, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [65536, 1024],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [1024, 65536],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    ([32128, 1024], "L_self_modules_embed_tokens_parameters_weight_"),
    ([1024], "L_self_modules_final_layer_norm_parameters_weight_"),
]
