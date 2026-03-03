from sympy import Symbol, Expr, Rel, Eq

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 80000}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_input_values_"),
    ([2], "L_self_modules_classifier_parameters_bias_"),
    ([2, 256], "L_self_modules_classifier_parameters_weight_"),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_k_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_q_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_attention_modules_v_proj_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_bias_",
    ),
    (
        [20, 16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_intermediate_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_bias_",
    ),
    (
        [16, 20],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_feed_forward_modules_output_dense_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_layers_modules_3_modules_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_0_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_10_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_10_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_11_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_11_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_12_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_12_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_13_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_13_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_14_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_14_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_15_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_15_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_1_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_2_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_3_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_4_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_5_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_6_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_6_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_7_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_7_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_8_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_8_modules_conv_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_9_modules_conv_parameters_bias_",
    ),
    (
        [16, 8, 19],
        "L_self_modules_data2vec_audio_modules_encoder_modules_pos_conv_embed_modules_layers_modules_9_modules_conv_parameters_weight_",
    ),
    (
        [32, 1, 8],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [32, 32, 8],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_1_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_1_modules_layer_norm_parameters_weight_",
    ),
    (
        [32, 32, 8],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_2_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_data2vec_audio_modules_feature_extractor_modules_conv_layers_modules_2_modules_layer_norm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_data2vec_audio_modules_feature_projection_modules_layer_norm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_data2vec_audio_modules_feature_projection_modules_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_data2vec_audio_modules_feature_projection_modules_projection_parameters_bias_",
    ),
    (
        [16, 32],
        "L_self_modules_data2vec_audio_modules_feature_projection_modules_projection_parameters_weight_",
    ),
    ([256], "L_self_modules_projector_parameters_bias_"),
    ([256, 16], "L_self_modules_projector_parameters_weight_"),
]
