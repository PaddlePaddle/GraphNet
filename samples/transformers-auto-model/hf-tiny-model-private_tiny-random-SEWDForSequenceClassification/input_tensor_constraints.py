from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 80000}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, S0], "L_input_values_"),
    ([2], "L_self_modules_classifier_parameters_bias_"),
    ([2, 256], "L_self_modules_classifier_parameters_weight_"),
    ([256], "L_self_modules_projector_parameters_bias_"),
    ([256, 32], "L_self_modules_projector_parameters_weight_"),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [20, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 20],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [20, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 20],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [20, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 20],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [20, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 20],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [512, 32],
        "L_self_modules_sew_d_modules_encoder_modules_encoder_modules_rel_embeddings_parameters_weight_",
    ),
    (
        [1, 1, 31],
        "L_self_modules_sew_d_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original0_",
    ),
    (
        [32, 16, 31],
        "L_self_modules_sew_d_modules_encoder_modules_pos_conv_embed_modules_conv_modules_parametrizations_modules_weight_parameters_original1_",
    ),
    (
        [32],
        "L_self_modules_sew_d_modules_encoder_modules_pos_conv_embed_modules_conv_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_sew_d_modules_encoder_modules_upsample_modules_projection_parameters_bias_",
    ),
    (
        [64, 32],
        "L_self_modules_sew_d_modules_encoder_modules_upsample_modules_projection_parameters_weight_",
    ),
    (
        [64, 1, 10],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_0_modules_layer_norm_parameters_weight_",
    ),
    (
        [32, 64, 3],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [32, 32, 1],
        "L_self_modules_sew_d_modules_feature_extractor_modules_conv_layers_modules_2_modules_conv_parameters_weight_",
    ),
    ([32], "L_self_modules_sew_d_modules_layer_norm_parameters_bias_"),
    ([32], "L_self_modules_sew_d_modules_layer_norm_parameters_weight_"),
]
