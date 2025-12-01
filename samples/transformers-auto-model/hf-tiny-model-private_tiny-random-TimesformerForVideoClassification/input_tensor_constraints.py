from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 10}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 2, 3, S0, S0], "L_pixel_values_"),
    ([10], "L_self_modules_classifier_parameters_bias_"),
    ([10, 32], "L_self_modules_classifier_parameters_weight_"),
    (
        [32],
        "L_self_modules_timesformer_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_",
    ),
    (
        [32, 3, 2, 2],
        "L_self_modules_timesformer_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_",
    ),
    ([1, 1, 32], "L_self_modules_timesformer_modules_embeddings_parameters_cls_token_"),
    (
        [1, 26, 32],
        "L_self_modules_timesformer_modules_embeddings_parameters_position_embeddings_",
    ),
    (
        [1, 2, 32],
        "L_self_modules_timesformer_modules_embeddings_parameters_time_embeddings_",
    ),
    (
        [96],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_qkv_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_qkv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_temporal_attention_modules_attention_modules_qkv_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_temporal_attention_modules_attention_modules_qkv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_temporal_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_temporal_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_temporal_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_temporal_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_temporal_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_0_modules_temporal_layernorm_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_qkv_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_qkv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_temporal_attention_modules_attention_modules_qkv_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_temporal_attention_modules_attention_modules_qkv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_temporal_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_temporal_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_temporal_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_temporal_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_temporal_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_1_modules_temporal_layernorm_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_qkv_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_qkv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_temporal_attention_modules_attention_modules_qkv_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_temporal_attention_modules_attention_modules_qkv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_temporal_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_temporal_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_temporal_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_temporal_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_temporal_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_2_modules_temporal_layernorm_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_qkv_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_qkv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_temporal_attention_modules_attention_modules_qkv_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_temporal_attention_modules_attention_modules_qkv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_temporal_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_temporal_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_temporal_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_temporal_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_temporal_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_3_modules_temporal_layernorm_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_qkv_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_qkv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_temporal_attention_modules_attention_modules_qkv_parameters_bias_",
    ),
    (
        [96, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_temporal_attention_modules_attention_modules_qkv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_temporal_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_temporal_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_temporal_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_temporal_dense_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_temporal_layernorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_timesformer_modules_encoder_modules_layer_modules_4_modules_temporal_layernorm_parameters_weight_",
    ),
    ([32], "L_self_modules_timesformer_modules_layernorm_parameters_bias_"),
    ([32], "L_self_modules_timesformer_modules_layernorm_parameters_weight_"),
]
