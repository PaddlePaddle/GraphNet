from sympy import Symbol, Expr, Rel, Eq


dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 43], "L_attention_mask_"),
    ([1, 43], "L_input_ids_"),
    ([768], "L_self_modules_embeddings_modules_LayerNorm_parameters_bias_"),
    ([768], "L_self_modules_embeddings_modules_LayerNorm_parameters_weight_"),
    (
        [6877, 768],
        "L_self_modules_embeddings_modules_word_embeddings_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_0_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_10_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_11_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_1_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_2_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_3_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_4_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_5_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_6_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_7_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_8_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_output_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_pos_key_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_pos_key_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_pos_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_pos_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_query_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_attention_modules_self_modules_value_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layer_modules_9_modules_output_modules_dense_parameters_weight_",
    ),
    ([1024, 768], "L_self_modules_encoder_modules_rel_embeddings_parameters_weight_"),
]
