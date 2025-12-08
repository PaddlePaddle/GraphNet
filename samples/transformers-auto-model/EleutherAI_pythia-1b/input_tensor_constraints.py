from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 2}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_attention_mask_"),
    ([S0, S1, 2048], "L_inputs_embeds_"),
    ([2048], "L_self_modules_final_layer_norm_parameters_bias_"),
    ([2048], "L_self_modules_final_layer_norm_parameters_weight_"),
    (
        [2048],
        "L_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_0_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_0_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_0_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_0_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_0_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_0_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_10_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_10_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_10_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_10_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_10_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_10_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_10_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_11_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_11_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_11_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_11_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_11_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_11_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_11_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_12_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_12_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_12_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_12_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_12_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_12_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_12_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_13_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_13_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_13_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_13_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_13_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_13_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_13_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_14_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_14_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_14_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_14_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_14_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_14_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_14_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_15_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_15_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_15_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_15_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_15_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_15_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_15_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_1_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_1_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_1_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_1_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_1_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_1_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_2_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_2_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_2_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_2_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_2_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_3_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_3_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_3_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_3_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_3_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_3_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_3_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_4_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_4_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_4_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_4_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_4_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_4_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_4_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_5_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_5_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_5_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_5_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_5_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_5_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_5_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_6_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_6_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_6_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_6_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_6_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_6_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_6_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_7_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_7_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_7_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_7_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_7_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_7_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_7_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_8_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_8_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_8_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_8_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_8_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_8_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_8_modules_post_attention_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_bias_",
    ),
    (
        [2048, 2048],
        "L_self_modules_layers_modules_9_modules_attention_modules_dense_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_bias_",
    ),
    (
        [6144, 2048],
        "L_self_modules_layers_modules_9_modules_attention_modules_query_key_value_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_9_modules_input_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_9_modules_input_layernorm_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_bias_",
    ),
    (
        [2048, 8192],
        "L_self_modules_layers_modules_9_modules_mlp_modules_dense_4h_to_h_parameters_weight_",
    ),
    (
        [8192],
        "L_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_bias_",
    ),
    (
        [8192, 2048],
        "L_self_modules_layers_modules_9_modules_mlp_modules_dense_h_to_4h_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layers_modules_9_modules_post_attention_layernorm_parameters_weight_",
    ),
    ([32], "L_self_modules_rotary_emb_buffers_inv_freq_"),
]
