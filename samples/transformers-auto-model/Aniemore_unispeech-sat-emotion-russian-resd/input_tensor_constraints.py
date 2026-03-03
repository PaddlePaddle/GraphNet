from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 80000}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_input_values_"),
    ([7], "L_self_modules_classifier_parameters_bias_"),
    ([7, 768], "L_self_modules_classifier_parameters_weight_"),
    ([768], "L_self_modules_projector_parameters_bias_"),
    ([768, 1024], "L_self_modules_projector_parameters_weight_"),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_10_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_11_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_12_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_13_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_14_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_15_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_16_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_17_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_18_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_19_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_20_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_21_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_22_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_23_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_4_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_5_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_6_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_7_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_8_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_layer_norm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_layers_modules_9_modules_layer_norm_parameters_weight_",
    ),
    (
        [1, 1, 128],
        "L_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original0_",
    ),
    (
        [1024, 64, 128],
        "L_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original1_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_encoder_modules_pos_conv_embed_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 10],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_1_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_2_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_3_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_3_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_4_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_4_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 2],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_5_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_5_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 2],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_6_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_6_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_extractor_modules_conv_layers_modules_6_modules_layer_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_unispeech_sat_modules_feature_projection_modules_layer_norm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_bias_",
    ),
    (
        [1024, 512],
        "L_self_modules_unispeech_sat_modules_feature_projection_modules_projection_parameters_weight_",
    ),
]
