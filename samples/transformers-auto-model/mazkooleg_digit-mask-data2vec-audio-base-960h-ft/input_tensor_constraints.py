dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 80000], "L_input_values_"),
    ([2], "L_self_modules_classifier_parameters_bias_"),
    ([2, 256], "L_self_modules_classifier_parameters_weight_"),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_10_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_11_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_4_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_5_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_6_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_7_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_8_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_9_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_0_modules_conv_parameters_bias_",
    ),
    (
        [768, 48, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_1_modules_conv_parameters_bias_",
    ),
    (
        [768, 48, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_2_modules_conv_parameters_bias_",
    ),
    (
        [768, 48, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_3_modules_conv_parameters_bias_",
    ),
    (
        [768, 48, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_4_modules_conv_parameters_bias_",
    ),
    (
        [768, 48, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [512, 1, 10],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_1_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_2_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_3_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_3_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_4_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_4_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 2],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_5_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_5_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 2],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_6_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_6_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_6_modules_layer_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_projection_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_data2vec_audio_modules_feature_projection_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_data2vec_audio_modules_feature_projection_modules_projection_parameters_bias_",
    ),
    (
        [768, 512],
        "L_self_modules_data2vec_audio_modules_feature_projection_modules_projection_parameters_weight_",
    ),
    ([256], "L_self_modules_projector_parameters_bias_"),
    ([256, 768], "L_self_modules_projector_parameters_weight_"),
]
