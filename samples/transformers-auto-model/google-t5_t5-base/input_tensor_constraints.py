dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 12], "L_encoder_attention_mask_"),
    ([1, 12, 768], "L_encoder_hidden_states_"),
    ([1, 12, 768], "L_inputs_embeds_"),
    (
        [768, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [32, 12],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_relative_attention_bias_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_0_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_0_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_0_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_0_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_10_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_10_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_10_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_10_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_10_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_11_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_11_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_11_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_11_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_11_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_1_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_1_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_1_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_1_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_2_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_2_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_2_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_2_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_3_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_3_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_3_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_3_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_4_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_4_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_4_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_4_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_4_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_5_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_5_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_5_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_5_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_5_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_6_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_6_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_6_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_6_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_6_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_7_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_7_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_7_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_7_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_7_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_8_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_8_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_8_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_8_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_8_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_SelfAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_9_modules_layer_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_k_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_o_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_q_parameters_weight_",
    ),
    (
        [768, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_EncDecAttention_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_9_modules_layer_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_block_modules_9_modules_layer_modules_2_modules_DenseReluDense_modules_wi_parameters_weight_",
    ),
    (
        [768, 3072],
        "L_self_modules_block_modules_9_modules_layer_modules_2_modules_DenseReluDense_modules_wo_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_block_modules_9_modules_layer_modules_2_modules_layer_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_final_layer_norm_parameters_weight_"),
]
