dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 51], "L_attention_mask_"),
    ([1, 51], "L_input_ids_"),
    ([1, 512], "L_self_modules_char_embeddings_buffers_position_ids_"),
    (
        [],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_0_norm_type",
    ),
    (
        [16384, 4],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_0_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_1_norm_type",
    ),
    (
        [16384, 4],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_1_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_2_norm_type",
    ),
    (
        [16384, 4],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_2_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_3_norm_type",
    ),
    (
        [16384, 4],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_3_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_4_norm_type",
    ),
    (
        [16384, 4],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_4_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_5_norm_type",
    ),
    (
        [16384, 4],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_5_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_6_norm_type",
    ),
    (
        [16384, 4],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_6_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_7_norm_type",
    ),
    (
        [16384, 4],
        "L_self_modules_char_embeddings_modules_HashBucketCodepointEmbedder_7_parameters_weight_",
    ),
    ([], "L_self_modules_char_embeddings_modules_LayerNorm_eps"),
    ([32], "L_self_modules_char_embeddings_modules_LayerNorm_parameters_bias_"),
    ([32], "L_self_modules_char_embeddings_modules_LayerNorm_parameters_weight_"),
    ([], "L_self_modules_char_embeddings_modules_char_position_embeddings_norm_type"),
    (
        [16384, 32],
        "L_self_modules_char_embeddings_modules_char_position_embeddings_parameters_weight_",
    ),
    ([], "L_self_modules_char_embeddings_modules_dropout_p"),
    ([], "L_self_modules_char_embeddings_modules_token_type_embeddings_norm_type"),
    (
        [16, 32],
        "L_self_modules_char_embeddings_modules_token_type_embeddings_parameters_weight_",
    ),
    ([], "L_self_modules_chars_to_molecules_modules_LayerNorm_eps"),
    ([32], "L_self_modules_chars_to_molecules_modules_LayerNorm_parameters_bias_"),
    ([32], "L_self_modules_chars_to_molecules_modules_LayerNorm_parameters_weight_"),
    ([32], "L_self_modules_chars_to_molecules_modules_conv_parameters_bias_"),
    ([32, 32, 4], "L_self_modules_chars_to_molecules_modules_conv_parameters_weight_"),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_dropout_p",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_dropout_p",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_dropout_p",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_dropout_p",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_dropout_p",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_dropout_p",
    ),
    (
        [32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_final_char_encoder_modules_layer_modules_0_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dropout_p",
    ),
    (
        [],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_dropout_p",
    ),
    (
        [32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_bias_",
    ),
    (
        [32, 32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_parameters_weight_",
    ),
    (
        [37],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [37, 32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_eps",
    ),
    (
        [32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [32, 37],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [],
        "L_self_modules_initial_char_encoder_modules_layer_modules_0_modules_output_modules_dropout_p",
    ),
    ([32], "L_self_modules_pooler_modules_dense_parameters_bias_"),
    ([32, 32], "L_self_modules_pooler_modules_dense_parameters_weight_"),
    ([], "L_self_modules_projection_modules_LayerNorm_eps"),
    ([32], "L_self_modules_projection_modules_LayerNorm_parameters_bias_"),
    ([32], "L_self_modules_projection_modules_LayerNorm_parameters_weight_"),
    ([32], "L_self_modules_projection_modules_conv_parameters_bias_"),
    ([32, 64, 4], "L_self_modules_projection_modules_conv_parameters_weight_"),
    ([], "L_self_modules_projection_modules_dropout_p"),
    ([1, 51], "L_token_type_ids_"),
]
