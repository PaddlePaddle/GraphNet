from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 80000}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_input_values_"),
    ([45], "L_self_modules_classifier_parameters_bias_"),
    ([45, 256], "L_self_modules_classifier_parameters_weight_"),
    ([256], "L_self_modules_projector_parameters_bias_"),
    ([256, 768], "L_self_modules_projector_parameters_weight_"),
    ([768], "L_self_modules_sew_modules_encoder_modules_layer_norm_parameters_bias_"),
    ([768], "L_self_modules_sew_modules_encoder_modules_layer_norm_parameters_weight_"),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_10_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_11_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_12_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_13_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_14_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_15_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_16_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_17_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_18_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_19_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_20_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_21_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_22_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_23_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_4_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_5_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_6_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_7_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_8_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_layers_modules_9_modules_layer_norm_parameters_weight_",
    ),
    (
        [1, 1, 31],
        "L_self_modules_sew_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original0_",
    ),
    (
        [768, 48, 31],
        "L_self_modules_sew_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original1_",
    ),
    (
        [768],
        "L_self_modules_sew_modules_encoder_modules_pos_conv_embed_modules_conv_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_sew_modules_encoder_modules_upsample_modules_projection_parameters_bias_",
    ),
    (
        [1536, 768],
        "L_self_modules_sew_modules_encoder_modules_upsample_modules_projection_parameters_weight_",
    ),
    (
        [64, 1, 10],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 1],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_10_modules_conv_parameters_weight_",
    ),
    (
        [512, 512, 2],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_11_modules_conv_parameters_weight_",
    ),
    (
        [512, 512, 1],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_12_modules_conv_parameters_weight_",
    ),
    (
        [128, 64, 3],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [128, 128, 1],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [128, 128, 3],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [128, 128, 1],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [256, 128, 3],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [256, 256, 1],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_6_modules_conv_parameters_weight_",
    ),
    (
        [256, 256, 3],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_7_modules_conv_parameters_weight_",
    ),
    (
        [256, 256, 1],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_8_modules_conv_parameters_weight_",
    ),
    (
        [512, 256, 2],
        "L_self_modules_sew_modules_feature_extractor_modules_conv_layers_modules_9_modules_conv_parameters_weight_",
    ),
    ([768], "L_self_modules_sew_modules_feature_projection_parameters_bias_"),
    ([768, 512], "L_self_modules_sew_modules_feature_projection_parameters_weight_"),
    ([512], "L_self_modules_sew_modules_layer_norm_parameters_bias_"),
    ([512], "L_self_modules_sew_modules_layer_norm_parameters_weight_"),
]
