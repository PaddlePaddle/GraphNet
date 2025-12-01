dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 80000], "L_reference_"),
    (
        [1],
        "L_self_modules_predictor_modules_att_pool_layer_modules_linear1_parameters_bias_",
    ),
    (
        [1, 64],
        "L_self_modules_predictor_modules_att_pool_layer_modules_linear1_parameters_weight_",
    ),
    (
        [5],
        "L_self_modules_predictor_modules_att_pool_layer_modules_linear2_parameters_bias_",
    ),
    (
        [5, 64],
        "L_self_modules_predictor_modules_att_pool_layer_modules_linear2_parameters_weight_",
    ),
    ([32], "L_self_modules_projector_parameters_bias_"),
    ([32, 768], "L_self_modules_projector_parameters_weight_"),
    (
        [512],
        "L_self_modules_ssl_model_modules_encoder_modules_feature_projection_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_ssl_model_modules_encoder_modules_feature_projection_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_feature_projection_modules_projection_parameters_bias_",
    ),
    (
        [768, 512],
        "L_self_modules_ssl_model_modules_encoder_modules_feature_projection_modules_projection_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_10_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_11_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_3_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_4_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_5_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_6_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_7_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_8_modules_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_layer_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_layers_modules_9_modules_layer_norm_parameters_weight_",
    ),
    (
        [1, 1, 128],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original0_",
    ),
    (
        [768, 48, 128],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original1_",
    ),
    (
        [768],
        "L_self_modules_ssl_model_modules_encoder_modules_transformer_modules_pos_conv_embed_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 10],
        "L_self_modules_ssl_model_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_ssl_model_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_ssl_model_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_ssl_model_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_ssl_model_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_ssl_model_modules_feature_extractor_modules_conv_layers_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [512, 512, 3],
        "L_self_modules_ssl_model_modules_feature_extractor_modules_conv_layers_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [512, 512, 2],
        "L_self_modules_ssl_model_modules_feature_extractor_modules_conv_layers_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [512, 512, 2],
        "L_self_modules_ssl_model_modules_feature_extractor_modules_conv_layers_modules_6_modules_conv_parameters_weight_",
    ),
    ([1, 80000], "L_waveform_"),
]
