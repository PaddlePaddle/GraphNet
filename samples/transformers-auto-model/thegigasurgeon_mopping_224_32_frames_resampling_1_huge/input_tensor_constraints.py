dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 32, 3, 224, 224], "L_pixel_values_"),
    ([2], "L_self_modules_classifier_parameters_bias_"),
    ([2, 1280], "L_self_modules_classifier_parameters_weight_"),
    ([1280], "L_self_modules_fc_norm_parameters_bias_"),
    ([1280], "L_self_modules_fc_norm_parameters_weight_"),
    (
        [1280],
        "L_self_modules_videomae_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_bias_",
    ),
    (
        [1280, 3, 2, 32, 32],
        "L_self_modules_videomae_modules_embeddings_modules_patch_embeddings_modules_projection_parameters_weight_",
    ),
    ([1, 784, 1280], "L_self_modules_videomae_modules_embeddings_position_embeddings"),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_12_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_13_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_14_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_15_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_16_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_17_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_18_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_19_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_20_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_21_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_22_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_23_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_24_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_25_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_26_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_27_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_28_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_29_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_30_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_31_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_key_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_query_parameters_weight_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_modules_value_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_parameters_q_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_parameters_v_bias_",
    ),
    (
        [],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_attention_modules_attention_scaling",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [5120, 1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_layernorm_after_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_layernorm_before_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1280, 5120],
        "L_self_modules_videomae_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
]
